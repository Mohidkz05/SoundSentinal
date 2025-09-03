# train_dp_avspoof.py

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
import torch.nn.functional as F
import argparse
# --- Hyperparameters & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# --- DP Parameters ---
MAX_GRAD_NORM = 1.0       # Clipping threshold for gradients
NOISE_MULTIPLIER = 1.1    # Amount of noise to add
TARGET_DELTA = 1e-5       # Target privacy parameter delta

# --- UPDATE THESE PATHS (relative & corpus-aware) ---
from pathlib import Path
import os

SCRIPT_DIR = Path(__file__).resolve().parent

def find_repo_root(start: Path) -> Path:
    cur = start
    while cur != cur.parent:
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start  # fallback

REPO_ROOT = find_repo_root(SCRIPT_DIR)

# Optional override: set ASVSPOOF_ROOT to point outside the repo if you keep data elsewhere
#   e.g.  set ASVSPOOF_ROOT=D:\datasets\ASVspoof2019
DATA_ROOT = Path(os.getenv("ASVSPOOF_ROOT", REPO_ROOT / "data"))

def get_corpus_paths(corpus: str = "LA"):
    corpus = corpus.upper()
    if corpus not in {"LA", "PA"}:
        raise ValueError("corpus must be 'LA' or 'PA'")

    # Folders
    train_audio = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_train" / "flac"
    dev_audio   = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_dev"   / "flac"
    proto_dir   = DATA_ROOT / corpus / f"ASVspoof2019_{corpus}_cm_protocols"

    # Protocol files (auto-pick by pattern)
    train_proto = next(proto_dir.glob(f"*{corpus}*cm*train*.*"))
    dev_proto   = next(proto_dir.glob(f"*{corpus}*cm*dev*.*"))

    # Sanity checks
    if not train_audio.exists(): raise FileNotFoundError(train_audio)
    if not dev_audio.exists():   raise FileNotFoundError(dev_audio)

    return {
        "TRAIN_AUDIO_DIR": train_audio,
        "DEV_AUDIO_DIR":   dev_audio,
        "TRAIN_PROTOCOL_FILE": train_proto,
        "DEV_PROTOCOL_FILE":   dev_proto,
        "REPO_ROOT": REPO_ROOT,
    }

# Choose which corpus to use here (or wire to argparse below)
PATHS = get_corpus_paths("LA")

TRAIN_AUDIO_DIR    = PATHS["TRAIN_AUDIO_DIR"]
DEV_AUDIO_DIR      = PATHS["DEV_AUDIO_DIR"]
TRAIN_PROTOCOL_FILE= PATHS["TRAIN_PROTOCOL_FILE"]
DEV_PROTOCOL_FILE  = PATHS["DEV_PROTOCOL_FILE"]

print("Repo root:", REPO_ROOT)
print("Train proto:", TRAIN_PROTOCOL_FILE)
print("Dev   proto:", DEV_PROTOCOL_FILE)
print("Train audio:", TRAIN_AUDIO_DIR)
print("Dev   audio:", DEV_AUDIO_DIR)

# --- Checkpointing ---
CKPT_DIR = (SCRIPT_DIR / "checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
LAST_CKPT = CKPT_DIR / "last.pth"
BEST_CKPT = CKPT_DIR / "best.pth"

def save_ckpt(model, optimizer, epoch, steps_done, is_best=False):
    """
    Save:
      - unwrapped model weights (model._module)
      - optimizer state
      - training progress
      - DP hyperparams (for reference)
    """
    payload = {
        "epoch": epoch,
        "steps_done": steps_done,
        "model": model._module.state_dict(),     # Opacus wraps -> save the underlying module
        "optimizer": optimizer.state_dict(),
        "dp": {
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "batch_size": BATCH_SIZE,
        },
    }
    # timestamped archive
    torch.save(payload, CKPT_DIR / f"deepfake_{strftime('%Y%m%d-%H%M%S')}.pth")
    # rolling "last"
    torch.save(payload, LAST_CKPT)
    if is_best:
        torch.save(payload, BEST_CKPT)



# ===================================================================
# 1. DATASET CLASS
# ===================================================================
class AVSpoofDataset(Dataset):
    def __init__(self, protocol_file, audio_dir, transform_fn, max_len=64000):
        self.protocol = pd.read_csv(protocol_file, delim_whitespace=True, header=None)

        self.protocol.columns = ['speaker_id', 'audio_file_name', '_', 'system_id', 'label']
        
        self.audio_dir = audio_dir
        self.transform_fn = transform_fn
        self.max_len = max_len
        
        self.label_map = {"bonafide": 0, "spoof": 1}

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        audio_name = self.protocol.iloc[idx]['audio_file_name']
        label_str = self.protocol.iloc[idx]['label']
        label = self.label_map[label_str]
        
        # Load audio waveform
        waveform, sample_rate = torchaudio.load(str(self.audio_dir / f"{audio_name}.flac"))
        
        # Pad or truncate the waveform to a fixed length
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        else:
            padding = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        # Apply the transformation (e.g., compute Mel spectrogram)
        spectrogram = self.transform_fn(waveform)
        
        return spectrogram, torch.tensor(label, dtype=torch.long)


# ===================================================================
# 2. MODEL ARCHITECTURE
# ===================================================================
class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super(AudioClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 128, 126)
            dummy_output = self.pool(F.relu(self.conv2(self.pool(F.relu(self.conv1(dummy_input))))))
            self.flattened_size = dummy_output.numel()

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ===================================================================
# 3. MAIN TRAINING FUNCTION
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="LA", choices=["LA","PA"])
    args = parser.parse_args()

    P = get_corpus_paths(args.corpus)
    TRAIN_AUDIO_DIR     = P["TRAIN_AUDIO_DIR"]
    DEV_AUDIO_DIR       = P["DEV_AUDIO_DIR"]
    TRAIN_PROTOCOL_FILE = P["TRAIN_PROTOCOL_FILE"]
    DEV_PROTOCOL_FILE   = P["DEV_PROTOCOL_FILE"]
    # Define the transformation to get Mel Spectrograms
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )

    # Create Dataset and DataLoader for train
    train_dataset = AVSpoofDataset(
        protocol_file=TRAIN_PROTOCOL_FILE,
        audio_dir=TRAIN_AUDIO_DIR,
        transform_fn=mel_spectrogram_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

    # Create Dataset and DataLoader for dev
    dev_dataset = AVSpoofDataset(
        protocol_file=DEV_PROTOCOL_FILE,
        audio_dir=DEV_AUDIO_DIR,
        transform_fn=mel_spectrogram_transform
    )
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)



    # Initialize Model, Optimizer, and Loss
    model = AudioClassifierCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Attach Opacus PrivacyEngine
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
    )

  # ---- Auto-resume if checkpoints/last.pth exists ----
    start_epoch, prev_steps = 1, 0
    if LAST_CKPT.exists():
        ckpt = torch.load(LAST_CKPT, map_location=DEVICE)
        # load weights into the unwrapped module
        model._module.load_state_dict(ckpt["model"])
        # restore optimizer (works because it's the wrapped DP optimizer we saved)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        prev_steps  = ckpt.get("steps_done", 0)
        print(f"Resumed from {LAST_CKPT} at epoch {start_epoch} (steps_done={prev_steps})")

    print("--- Starting Differentially Private Training ---")
    
    # Training Loop
    steps_done = prev_steps
    best_loss = float("inf")

    steps_done = prev_steps
    best_eer = float("inf")     # ← track best dev EER
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for inputs, labels in progress_bar:
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
                'Loss': f'{total_loss / (total / BATCH_SIZE):.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
    # ---- end of epoch: report train, then evaluate on DEV ----
    train_avg_loss = total_loss / len(train_loader)
    epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)
    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"[TRAIN] loss={train_avg_loss:.4f} acc={100*correct/total:.2f}% "
        f"(ε={epsilon:.2f}, δ={TARGET_DELTA})"
    )

    # Evaluate on dev (no gradients)
    dev_loss, dev_acc, dev_eer, dev_thresh = evaluate(model, dev_loader, criterion, DEVICE)
    print(f"[DEV]   loss={dev_loss:.4f} acc={dev_acc*100:.2f}% EER={dev_eer*100:.2f}% (thr={dev_thresh:.4f})")

    # Save checkpoints — choose best by DEV EER
    is_best = dev_eer < best_eer
    if is_best:
        best_eer = dev_eer
    save_ckpt(model, optimizer, epoch, steps_done, is_best=is_best)


    print("\n--- Training Finished ---")

    # ===================================================================
    # 4. SAVE THE TRAINED MODEL (NEW PART) 💾
    # ===================================================================
    MODEL_SAVE_PATH = "deepfake_audio_detector.pth"
    
    # We save the model's learned weights (state_dict)
    # Opacus wraps the model, so we access the original model via ._module
    torch.save(model._module.state_dict(), MODEL_SAVE_PATH)
    
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")


def compute_eer_np(labels, scores):
    """labels: 0=bonafide, 1=spoof; scores: higher means more 'spoof'."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    idx = np.argsort(scores)[::-1]     # sort by score desc
    scores, labels = scores[idx], labels[idx]
    P = (labels == 1).sum()
    N = (labels == 0).sum()
    tp = np.cumsum(labels == 1)
    fp = np.cumsum(labels == 0)
    fn = P - tp
    tn = N - fp
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[i] + fnr[i]) / 2)
    thresh = float(scores[i])
    return eer, thresh

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

        # accuracy
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        # spoof score = prob of class 1
        probs = F.softmax(out, dim=1)
        all_scores.append(probs[:, 1].detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / max(1, batches)
    acc = correct / max(1, total)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    eer, thresh = compute_eer_np(labels, scores)
    return avg_loss, acc, eer, thresh

if __name__ == '__main__':
    main()