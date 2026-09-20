# train_dp_avspoof.py (Final Sturdy Version)

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from tqdm import tqdm
from pathlib import Path
from time import strftime
import numpy as np
import argparse
import os

# Model + preprocessing live in model.py so app.py serves exactly what we train.
from model import (
    ARCHITECTURES,
    DEFAULT_ARCH,
    DEFAULT_FRONTEND,
    FRONTENDS,
    LABEL_MAP,
    MAX_LEN,
    SAMPLE_RATE,
    build_model,
    build_transform,
    check_pairing,
    default_frontend_for,
    load_audio,
    preprocess_waveform,
)

# --- Hyperparameters & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Each architecture trains under its own published recipe, overridable from the
# command line. Holding the schedule fixed across architectures would be the
# cleaner experiment and it is not available to us: AASIST cannot run at batch
# 64. Its first residual block produces a (batch, 32, 24, 21290) activation,
# which is 4.2 GB at batch 64 before the backward pass stores anything — so the
# batch size is a hardware fact, not a choice, and the published 24 is used.
# The learning rate follows for the same reason: 1e-3 is tuned for the CNN and
# the AASIST authors use 1e-4 with weight decay and cosine annealing.
#
# The consequence must be stated wherever these rows are compared: the AASIST
# row differs from the CNN rows by architecture AND schedule, so the gap is not
# attributable to architecture alone. See RESULTS.md.
TRAIN_DEFAULTS = {
    #          epochs  batch  lr      weight decay  cosine floor
    "cnn":      {"epochs": 5,   "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0,  "lr_min": None},
    "aasist":   {"epochs": 100, "batch_size": 24, "lr": 1e-4, "weight_decay": 1e-4, "lr_min": 5e-6},
    "aasist-l": {"epochs": 100, "batch_size": 24, "lr": 1e-4, "weight_decay": 1e-4, "lr_min": 5e-6},
}

# --- NEW: Reproducibility ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# For CUDA
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- DP Parameters ---
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.1
TARGET_DELTA = 1e-5

# --- Professional Project Structure ---
SCRIPT_DIR = Path(__file__).resolve().parent

def find_repo_root(start: Path) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start

REPO_ROOT = find_repo_root(SCRIPT_DIR)
DATA_ROOT = Path(os.getenv("ASVSPOOF_ROOT", REPO_ROOT / "data"))

def get_corpus_paths(corpus: str = "LA"):
    corpus = corpus.upper()
    if corpus not in {"LA", "PA"}:
        raise ValueError("corpus must be 'LA' or 'PA'")
    train_audio = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_train" / "flac"
    dev_audio   = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_dev"   / "flac"
    proto_dir   = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_cm_protocols"
    train_proto = next(proto_dir.glob(f"*{corpus}*cm*train*.*"))
    dev_proto   = next(proto_dir.glob(f"*{corpus}*cm*dev*.*"))
    if not train_audio.exists(): raise FileNotFoundError(train_audio)
    if not dev_audio.exists():   raise FileNotFoundError(dev_audio)
    paths = {
        "TRAIN_AUDIO_DIR": train_audio, "DEV_AUDIO_DIR": dev_audio,
        "TRAIN_PROTOCOL_FILE": train_proto, "DEV_PROTOCOL_FILE": dev_proto,
        "REPO_ROOT": REPO_ROOT,
    }
    # Eval is resolved but never required. Training does not touch it — the whole
    # point of the partition is that the model never sees A07–A19 — and a corpus
    # copy without it must still be trainable, so a missing eval tree is absent
    # from the dict rather than an error. evaluate.py is what insists on it.
    eval_audio = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_eval" / "flac"
    eval_protos = sorted(proto_dir.glob(f"*{corpus}*cm*eval*.*"))
    if eval_audio.exists() and eval_protos:
        paths["EVAL_AUDIO_DIR"] = eval_audio
        paths["EVAL_PROTOCOL_FILE"] = eval_protos[0]
    # The organisers' ASV scores, needed for min t-DCF. Shipped with the corpus
    # and deliberately not regenerated: t-DCF only compares across systems
    # because every system is scored against the same ASV.
    for part in ("dev", "eval"):
        asv = sorted((DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_asv_scores")
                     .glob(f"*{part}*scores*")) if (
                         DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_asv_scores").exists() else []
        if asv:
            paths[f"{part.upper()}_ASV_SCORES"] = asv[0]
    return paths

# --- Smart Checkpointing ---
# DP and baseline runs get separate directories. They share an architecture but
# not a training regime, so a shared last.pth would let one run auto-resume from
# the other's weights. app.py reads checkpoints/best.pth, so DP keeps the root.
# $CKPT_ROOT relocates the checkpoint tree off the repo. On an HPC account the
# clone sits in a small home quota while runs belong in project storage; app.py
# reads the same variable so the server still finds best.pth.
CKPT_DIR = Path(os.getenv("CKPT_ROOT", SCRIPT_DIR / "checkpoints"))

def get_ckpt_paths(use_dp: bool, frontend: str = DEFAULT_FRONTEND, arch: str = DEFAULT_ARCH):
    """Each (architecture, front-end, privacy regime) triple gets its own directory.

    Same reasoning as the DP/non-DP split above, one level out: a log-Mel and an
    LFCC run share an architecture but not an input space, so a shared last.pth
    would let one silently auto-resume from the other's weights and the
    resulting numbers would be unattributable. Two architectures do not even
    share a state_dict, so mixing those would fail loudly rather than quietly —
    but it would still cost a run. The default architecture and front-end keep
    the original layout so existing checkpoints stay where app.py looks.
    """
    ckpt_dir = CKPT_DIR if arch == DEFAULT_ARCH else CKPT_DIR / arch
    if frontend != default_frontend_for(arch):
        ckpt_dir = ckpt_dir / frontend
    if not use_dp:
        ckpt_dir = ckpt_dir / "nodp"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, ckpt_dir / "last.pth", ckpt_dir / "best.pth"

def unwrap(model):
    """Opacus wraps the module in a GradSampleModule; --no-dp runs have no wrapper."""
    return getattr(model, "_module", model)

def save_ckpt(model, optimizer, epoch, steps_done, paths, use_dp, class_weights=None,
              is_best=False, metrics=None, best_eer=None, frontend=DEFAULT_FRONTEND,
              arch=DEFAULT_ARCH, batch_size=None, scheduler=None):
    """Write the rolling, best and timestamped checkpoints.

    `metrics` carries the dev-set numbers for this epoch, and with them the
    calibrated operating point. That threshold is the whole reason this
    argument exists: compute_eer_np has always returned it and it used to be
    printed and dropped on the floor, so the server fell back to an implicit
    0.5 cutoff and the tuned operating point never reached production. It is
    the EER point on P(spoof), so serving means `spoof if p >= threshold`.

    `best_eer` is stored so a resumed run knows what it is trying to beat.
    Without it best.pth was overwritten by whatever the first epoch after a
    resume produced, however much worse it was.
    """
    ckpt_dir, last_ckpt, best_ckpt = paths
    payload = {
        "epoch": epoch, "steps_done": steps_done,
        # Which features this model was trained on. app.py and evaluate.py read
        # it back and build the matching transform; without it, serving a
        # log-Mel pipeline to an LFCC-trained model is a silent wrong answer.
        "frontend": frontend,
        # And which network. Loading AASIST weights into the CNN raises, so this
        # one fails loudly rather than silently — but only if something knows to
        # build the right class first, which is what this field is for.
        "arch": arch,
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        # Restored on resume so a cosine schedule survives a job hitting its
        # Slurm time limit, which at 100 epochs it is expected to.
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "batch_size": batch_size,
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "metrics": metrics,
        # Promoted out of `metrics` because this is the one value the server
        # needs, and it should not have to know the shape of an eval record.
        "threshold": None if metrics is None else metrics.get("dev_threshold"),
        "best_eer": best_eer,
        "dp": {
            "noise_multiplier": NOISE_MULTIPLIER, "max_grad_norm": MAX_GRAD_NORM,
            "batch_size": batch_size,
        } if use_dp else None,
    }
    torch.save(payload, ckpt_dir / f"deepfake_{strftime('%Y%m%d-%H%M%S')}.pth")
    torch.save(payload, last_ckpt)
    if is_best:
        torch.save(payload, best_ckpt)
        print(f"🎉 New best model saved to {best_ckpt}!")

# ===================================================================
# 1. DATASET CLASS
# ===================================================================
class AVSpoofDataset(Dataset):
    def __init__(self, protocol_file, audio_dir, transform_pipeline, target_sample_rate=SAMPLE_RATE, max_len=MAX_LEN):
        self.protocol = pd.read_csv(protocol_file, sep=r'\s+', header=None, engine='python')
        self.protocol.columns = ['speaker_id', 'audio_file_name', '_', 'system_id', 'label']
        self.audio_dir = audio_dir
        self.transform_pipeline = transform_pipeline
        self.max_len = max_len
        self.target_sample_rate = target_sample_rate
        self.label_map = LABEL_MAP

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        audio_name = self.protocol.iloc[idx]['audio_file_name']
        label_str = self.protocol.iloc[idx]['label']
        label = self.label_map[label_str]

        waveform, sample_rate = load_audio(str(self.audio_dir / f"{audio_name}.flac"))

        # Same preprocessing the Flask server applies at inference time.
        spectrogram = preprocess_waveform(
            waveform, sample_rate, self.transform_pipeline, max_len=self.max_len
        )

        return spectrogram, torch.tensor(label, dtype=torch.long)

# ===================================================================
# 2. EVALUATION AND EER
# ===================================================================
def compute_eer_np(labels, scores):
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores, dtype=np.float64)
    idx = np.argsort(scores)[::-1]
    scores, labels = scores[idx], labels[idx]
    P, N = (labels == 1).sum(), (labels == 0).sum()
    tp, fp = np.cumsum(labels == 1), np.cumsum(labels == 0)
    fn, tn = P - tp, N - fp
    fpr, fnr = fp / (fp + tn + 1e-12), fn / (fn + tp + 1e-12)
    i = np.nanargmin(np.abs(fnr - fpr))
    eer, thresh = float((fpr[i] + fnr[i]) / 2), float(scores[i])
    return eer, thresh

def compute_class_weights(dataset, device):
    """Inverse-frequency weights from the actual protocol counts.

    LA train is ~1:9 bonafide:spoof, so an unweighted loss drifts toward calling
    everything spoof while still looking accurate. Weight the loss rather than
    using WeightedRandomSampler: Opacus's make_private replaces the loader's
    sampler with Poisson sampling, so a custom sampler is silently discarded.
    """
    counts = dataset.protocol['label'].map(LABEL_MAP).value_counts()
    n_classes = len(LABEL_MAP)
    total = int(counts.sum())
    weights = torch.tensor(
        [total / (n_classes * max(1, int(counts.get(i, 0)))) for i in range(n_classes)],
        dtype=torch.float32, device=device,
    )
    counts_str = ", ".join(f"{name}={int(counts.get(i, 0))}" for i, name in
                           enumerate(sorted(LABEL_MAP, key=LABEL_MAP.get)))
    print(f"Class counts: {counts_str} -> weights {weights.tolist()}")
    return weights

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total, correct, batches = 0.0, 0, 0, 0
    all_scores, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        batches += 1
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        probs = F.softmax(out, dim=1)
        all_scores.append(probs[:, 1].detach().cpu())
        all_labels.append(y.detach().cpu())
    avg_loss = total_loss / max(1, batches)
    acc = correct / max(1, total)
    scores, labels = torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()
    eer, thresh = compute_eer_np(labels, scores)

    # Confusion matrix at the calibrated threshold, not at argmax. On 1:9 data
    # "90% accurate" can mean "always guesses spoof", and only these four
    # numbers show which of the two it is.
    pred = (scores >= thresh).astype(int)
    cm = {
        "tn": int(((labels == 0) & (pred == 0)).sum()),
        "fp": int(((labels == 0) & (pred == 1)).sum()),
        "fn": int(((labels == 1) & (pred == 0)).sum()),
        "tp": int(((labels == 1) & (pred == 1)).sum()),
    }
    return avg_loss, acc, eer, thresh, cm

# ===================================================================
# 3. MAIN TRAINING AND EVALUATION FUNCTION
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="DP Deepfake Audio Trainer")
    parser.add_argument("--corpus", default="LA", choices=["LA", "PA"], help="ASVspoof corpus to use.")
    parser.add_argument("--no-dp", dest="use_dp", action="store_false",
                        help="Skip Opacus make_private and train a non-private baseline. "
                             "The baseline is the accuracy ceiling; the gap to a DP run is "
                             "the measured cost of privacy. Checkpoints go to checkpoints/nodp/.")
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false",
                        help="Disable inverse-frequency class weighting (for ablation).")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=list(ARCHITECTURES),
                        help="Network. 'cnn' is the 2-conv baseline reading a "
                             "spectrogram. 'aasist' reads the raw waveform through "
                             "learnable band-pass filters and relates distant parts "
                             "of the clip through a spectro-temporal graph; it is "
                             "the answer to RESULTS.md Finding 2, where log-Mel and "
                             "LFCC each won on attacks the other missed, so no "
                             "handcrafted front-end was the right one. 'aasist-l' is "
                             "the 85k-parameter variant, the fallback if memory is "
                             "tight. Checkpoints go to checkpoints/<arch>/.")
    parser.add_argument("--frontend", default=None, choices=list(FRONTENDS),
                        help="Front-end, defaulting to the one the architecture is "
                             "built to read (raw for AASIST, log-Mel for the CNN). "
                             "For the CNN, 'lfcc' is the controlled ablation against "
                             "'logmel': the Mel scale compresses high frequencies, "
                             "where vocoder artefacts live, and both official "
                             "ASVspoof baselines are cepstral.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override the architecture's default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the architecture's default batch size. Raising "
                             "it for AASIST is how you run out of GPU memory.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the architecture's default learning rate.")
    parser.add_argument("--num-workers", type=int,
                        default=int(os.getenv("SLURM_CPUS_PER_TASK", "2")),
                        help="DataLoader workers. Defaults to $SLURM_CPUS_PER_TASK inside a "
                             "Slurm job, else 2. At this model size the dataloader is the "
                             "bottleneck, not the GPU — but every worker forks the process, "
                             "so RAM caps it (keep it at 8 on the 6.7GB WSL box).")
    args = parser.parse_args()

    # The front-end follows the architecture unless it is named explicitly, and
    # an impossible pairing stops the run here rather than training on nonsense
    # for an hour — see check_pairing in model.py.
    frontend = args.frontend or default_frontend_for(args.arch)
    check_pairing(args.arch, frontend)

    recipe = dict(TRAIN_DEFAULTS[args.arch])
    if args.epochs is not None:
        recipe["epochs"] = args.epochs
    if args.batch_size is not None:
        recipe["batch_size"] = args.batch_size
    if args.lr is not None:
        recipe["lr"] = args.lr
    epochs, batch_size = recipe["epochs"], recipe["batch_size"]

    # Announce the device. A Slurm job that fell back to CPU because --gres was
    # missing is indistinguishable from a slow one until you read this line.
    if DEVICE.type == "cuda":
        print(f"Device: cuda -> {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB), "
              f"torch {torch.__version__}")
    else:
        print(f"Device: CPU (no CUDA visible), torch {torch.__version__}")
    print(f"DataLoader workers: {args.num_workers}")

    PATHS = get_corpus_paths(args.corpus)
    print(f"--- Using Corpus: {args.corpus} ---")
    for key, val in PATHS.items(): print(f"{key}: {val}")

    # Preprocessing shared with the inference server (see model.py). For the
    # raw front-end this is Identity — AASIST's SincConv is the transform.
    transform_pipeline = build_transform(frontend)
    print(f"Architecture: {args.arch}, front-end: {frontend}")
    print(f"Recipe: {epochs} epochs, batch {batch_size}, lr {recipe['lr']}, "
          f"weight decay {recipe['weight_decay']}, "
          f"cosine floor {recipe['lr_min'] or 'none (constant lr)'}")

    train_dataset = AVSpoofDataset(PATHS["TRAIN_PROTOCOL_FILE"], PATHS["TRAIN_AUDIO_DIR"], transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    dev_dataset = AVSpoofDataset(PATHS["DEV_PROTOCOL_FILE"], PATHS["DEV_AUDIO_DIR"], transform_pipeline)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.arch).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    optimizer = optim.Adam(model.parameters(), lr=recipe["lr"],
                           weight_decay=recipe["weight_decay"])
    # Cosine annealing is part of AASIST's published recipe and the CNN rows
    # never had it, so it is per-architecture rather than global. Its state is
    # checkpointed: at 100 epochs a job WILL hit its Slurm time limit, and a
    # resume that restarted the schedule at its peak would undo the anneal.
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                             eta_min=recipe["lr_min"])
        if recipe["lr_min"] else None
    )
    class_weights = compute_class_weights(train_dataset, DEVICE) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    privacy_engine = None
    if args.use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM,
        )

    paths = get_ckpt_paths(args.use_dp, frontend, args.arch)
    _, LAST_CKPT, _ = paths

    start_epoch, prev_steps, best_eer = 1, 0, float("inf")
    if LAST_CKPT.exists():
        print(f"Resuming from checkpoint: {LAST_CKPT}")
        # weights_only=False for the same reason as app.py and evaluate.py: a
        # DP checkpoint carries a numpy scalar from get_epsilon(), and torch
        # 2.6+ refuses those by default. Without this, auto-resume — which a
        # 100-epoch AASIST run depends on to survive its Slurm time limit —
        # dies on its own last.pth.
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE, weights_only=False)
        # Before load_state_dict, not after: loading the wrong architecture does
        # fail, but with two hundred lines of missing and unexpected keys, which
        # buries the one fact that explains it. get_ckpt_paths keeps the
        # architectures apart, so reaching this means $CKPT_ROOT points
        # somewhere it should not.
        ckpt_arch = ckpt.get("arch", DEFAULT_ARCH)
        if ckpt_arch != args.arch:
            raise SystemExit(
                f"ERROR: {LAST_CKPT} holds a {ckpt_arch!r} model, but this run is "
                f"{args.arch!r}. Architectures get their own checkpoint directory — "
                f"check $CKPT_ROOT, which is currently {CKPT_DIR}."
            )
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        prev_steps  = ckpt.get("steps_done", 0)
        # Older checkpoints predate this field; inf means the next epoch wins,
        # which is the old behaviour rather than a new failure mode.
        best_eer = ckpt.get("best_eer") or float("inf")
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])

    # --- NEW: Check if training is already complete ---
    if start_epoch > epochs:
        print(f"✅ Training already completed for {epochs}/{epochs} epochs. Exiting.")
        return
    # ------------------------------------------------

    mode = ("Differentially Private" if args.use_dp else "Non-Private Baseline (--no-dp)")
    mode = f"{mode}, {args.arch} on {frontend}"
    print(f"--- Starting {mode} Training ---")
    steps_done = prev_steps
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        # disable=None is tqdm's "off unless stderr is a terminal" — inside a Slurm
        # job the bar would otherwise write one line per update into the log file.
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=None)
        for batch_idx, (inputs, labels) in enumerate(progress_bar, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            steps_done += 1
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix({
                'Loss': f'{total_loss / batch_idx:.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        train_avg_loss = total_loss / len(train_loader)
        if privacy_engine is not None:
            epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)
            privacy_str = f" (ε={epsilon:.2f}, δ={TARGET_DELTA})"
        else:
            privacy_str = " (no DP)"
        print(
            f"Epoch {epoch}/{epochs} | "
            f"[TRAIN] loss={train_avg_loss:.4f} acc={100*correct/total:.2f}%"
            f"{privacy_str}"
        )

        dev_loss, dev_acc, dev_eer, dev_thresh, dev_cm = evaluate(model, dev_loader, criterion, DEVICE)
        print(f"[DEV]   loss={dev_loss:.4f} acc={dev_acc*100:.2f}% EER={dev_eer*100:.2f}% (thr={dev_thresh:.4f})")
        print(f"[DEV]   confusion @thr: bonafide {dev_cm['tn']} ok / {dev_cm['fp']} flagged | "
              f"spoof {dev_cm['tp']} caught / {dev_cm['fn']} missed")

        is_best = dev_eer < best_eer
        if is_best:
            best_eer = dev_eer
        metrics = {
            "dev_loss": dev_loss, "dev_acc": dev_acc, "dev_eer": dev_eer,
            "dev_threshold": dev_thresh, "dev_confusion": dev_cm,
            "train_loss": train_avg_loss, "train_acc": correct / max(1, total),
            # float(), because Opacus returns a numpy scalar and a checkpoint
            # holding one cannot be read back under torch's weights_only
            # default. The loaders pass weights_only=False regardless, but new
            # checkpoints should not need them to.
            "epsilon": float(epsilon) if privacy_engine is not None else None,
            "delta": TARGET_DELTA if privacy_engine is not None else None,
            "corpus": args.corpus,
            "frontend": frontend,
            "arch": args.arch,
            "lr": optimizer.param_groups[0]["lr"],
        }
        save_ckpt(model, optimizer, epoch, steps_done, paths, args.use_dp,
                  class_weights=class_weights, is_best=is_best,
                  metrics=metrics, best_eer=best_eer, frontend=frontend,
                  arch=args.arch, batch_size=batch_size, scheduler=scheduler)
        if scheduler is not None:
            scheduler.step()

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()