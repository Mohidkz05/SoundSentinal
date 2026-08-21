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
    AudioClassifierCNN,
    LABEL_MAP,
    MAX_LEN,
    SAMPLE_RATE,
    build_transform,
    load_audio,
    preprocess_waveform,
)

# --- Hyperparameters & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

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
    return {
        "TRAIN_AUDIO_DIR": train_audio, "DEV_AUDIO_DIR": dev_audio,
        "TRAIN_PROTOCOL_FILE": train_proto, "DEV_PROTOCOL_FILE": dev_proto,
        "REPO_ROOT": REPO_ROOT,
    }

# --- Smart Checkpointing ---
# DP and baseline runs get separate directories. They share an architecture but
# not a training regime, so a shared last.pth would let one run auto-resume from
# the other's weights. app.py reads checkpoints/best.pth, so DP keeps the root.
CKPT_DIR = (SCRIPT_DIR / "checkpoints")

def get_ckpt_paths(use_dp: bool):
    ckpt_dir = CKPT_DIR if use_dp else CKPT_DIR / "nodp"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, ckpt_dir / "last.pth", ckpt_dir / "best.pth"

def unwrap(model):
    """Opacus wraps the module in a GradSampleModule; --no-dp runs have no wrapper."""
    return getattr(model, "_module", model)

def save_ckpt(model, optimizer, epoch, steps_done, paths, use_dp, class_weights=None,
              is_best=False, metrics=None, best_eer=None):
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
        "model": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_size": BATCH_SIZE,
        "class_weights": None if class_weights is None else class_weights.tolist(),
        "metrics": metrics,
        # Promoted out of `metrics` because this is the one value the server
        # needs, and it should not have to know the shape of an eval record.
        "threshold": None if metrics is None else metrics.get("dev_threshold"),
        "best_eer": best_eer,
        "dp": {
            "noise_multiplier": NOISE_MULTIPLIER, "max_grad_norm": MAX_GRAD_NORM,
            "batch_size": BATCH_SIZE,
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
    args = parser.parse_args()

    PATHS = get_corpus_paths(args.corpus)
    print(f"--- Using Corpus: {args.corpus} ---")
    for key, val in PATHS.items(): print(f"{key}: {val}")

    # Log-Mel pipeline, shared with the inference server (see model.py).
    transform_pipeline = build_transform()

    train_dataset = AVSpoofDataset(PATHS["TRAIN_PROTOCOL_FILE"], PATHS["TRAIN_AUDIO_DIR"], transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_dataset = AVSpoofDataset(PATHS["DEV_PROTOCOL_FILE"], PATHS["DEV_AUDIO_DIR"], transform_pipeline)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = AudioClassifierCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    class_weights = compute_class_weights(train_dataset, DEVICE) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    privacy_engine = None
    if args.use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM,
        )

    paths = get_ckpt_paths(args.use_dp)
    _, LAST_CKPT, _ = paths

    start_epoch, prev_steps, best_eer = 1, 0, float("inf")
    if LAST_CKPT.exists():
        print(f"Resuming from checkpoint: {LAST_CKPT}")
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        unwrap(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        prev_steps  = ckpt.get("steps_done", 0)
        # Older checkpoints predate this field; inf means the next epoch wins,
        # which is the old behaviour rather than a new failure mode.
        best_eer = ckpt.get("best_eer") or float("inf")

    # --- NEW: Check if training is already complete ---
    if start_epoch > EPOCHS:
        print(f"✅ Training already completed for {EPOCHS}/{EPOCHS} epochs. Exiting.")
        return
    # ------------------------------------------------

    mode = "Differentially Private" if args.use_dp else "Non-Private Baseline (--no-dp)"
    print(f"--- Starting {mode} Training ---")
    steps_done = prev_steps
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
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
            f"Epoch {epoch}/{EPOCHS} | "
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
            "epsilon": epsilon if privacy_engine is not None else None,
            "delta": TARGET_DELTA if privacy_engine is not None else None,
            "corpus": args.corpus,
        }
        save_ckpt(model, optimizer, epoch, steps_done, paths, args.use_dp,
                  class_weights=class_weights, is_best=is_best,
                  metrics=metrics, best_eer=best_eer)

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()