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

# --- Hyperparameters & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

# --- DP Parameters ---
MAX_GRAD_NORM = 1.0       # Clipping threshold for gradients
NOISE_MULTIPLIER = 1.1    # Amount of noise to add
TARGET_DELTA = 1e-5       # Target privacy parameter delta

# --- UPDATE THESE PATHS ---
# This setup assumes your audio files and text file are in the same folder as the script.
TRAIN_PROTOCOL_FILE = "train_file.txt" # The text file with audio names and labels
TRAIN_AUDIO_DIR = "."                  # "." means the current directory


# ===================================================================
# 1. DATASET CLASS
# ===================================================================
class AVSpoofDataset(Dataset):
    def __init__(self, protocol_file, audio_dir, transform_fn, max_len=64000):
        self.protocol = pd.read_csv(protocol_file, sep=" ", header=None)
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
        waveform, sample_rate = torchaudio.load(f"{self.audio_dir}/{audio_name}.flac")
        
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
    # Define the transformation to get Mel Spectrograms
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )

    # Create Dataset and DataLoader
    train_dataset = AVSpoofDataset(
        protocol_file=TRAIN_PROTOCOL_FILE,
        audio_dir=TRAIN_AUDIO_DIR,
        transform_fn=mel_spectrogram_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )

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

    print("--- Starting Differentially Private Training ---")
    
    # Training Loop
    for epoch in range(1, EPOCHS + 1):
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
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{total_loss / (total / BATCH_SIZE):.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        epsilon = privacy_engine.get_epsilon(delta=TARGET_DELTA)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Train Accuracy: {100 * correct / total:.2f}% | "
            f"(ε = {epsilon:.2f}, δ = {TARGET_DELTA})"
        )

    print("\n--- Training Finished ---")

    # ===================================================================
    # 4. SAVE THE TRAINED MODEL (NEW PART) 💾
    # ===================================================================
    MODEL_SAVE_PATH = "deepfake_audio_detector.pth"
    
    # We save the model's learned weights (state_dict)
    # Opacus wraps the model, so we access the original model via ._module
    torch.save(model._module.state_dict(), MODEL_SAVE_PATH)
    
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()