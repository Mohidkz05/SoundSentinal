# model.py
#
# Single source of truth for the network and the audio preprocessing.
#
# Both train_dp_avspoof.py and app.py import from here. They used to each keep
# their own copy of the CNN and their own preprocessing steps, which silently
# drifted apart (the trainer moved to log-Mel + standardization while the server
# stayed on raw power Mel). Keep new preprocessing changes in this file so the
# two sides cannot disagree again.

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

# --- Audio constants (shared by training and inference) ---
SAMPLE_RATE = 16000
MAX_LEN = 64000          # 4 seconds at 16 kHz
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TOP_DB = 80


def load_audio(path_or_file):
    """
    Read an audio file into a (channels, frames) float32 tensor, plus its rate.

    Uses soundfile rather than torchaudio.load: as of torchaudio 2.11 that call
    delegates to TorchCodec, which needs system FFmpeg libraries installed.
    soundfile bundles libsndfile in the wheel, so FLAC/WAV work everywhere with
    no system packages. Accepts a path or an open file object (Flask uploads).
    """
    data, sample_rate = sf.read(path_or_file, dtype="float32", always_2d=True)
    # soundfile gives (frames, channels); torch convention is (channels, frames).
    waveform = torch.from_numpy(data).T.contiguous()
    return waveform, sample_rate


def build_transform():
    """Log-Mel spectrogram pipeline. Use this everywhere audio becomes a tensor."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        ),
        T.AmplitudeToDB(stype="power", top_db=TOP_DB),
    )


def preprocess_waveform(waveform, sample_rate, transform_pipeline, max_len=MAX_LEN):
    """
    Waveform -> standardized log-Mel spectrogram, shape (1, N_MELS, frames).

    Downmix to mono, resample to SAMPLE_RATE, pad/truncate to max_len, apply the
    transform, then standardize per sample.
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        padding = max_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, padding))

    spectrogram = transform_pipeline(waveform)

    # Per-sample standardization
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)

    return spectrogram


# ===================================================================
# MODEL ARCHITECTURE
# ===================================================================
class AudioClassifierCNN(nn.Module):
    def __init__(self):
        super(AudioClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Adaptive pooling instead of a hardcoded flattened size, so the model
        # tolerates changes to the input spectrogram dimensions.
        self.gap = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(32 * 8 * 8, 128)

        self.fc2 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = self.gap(x)
        x = x.flatten(1)  # Flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Label order is fixed by the training protocol: bonafide=0, spoof=1.
LABEL_MAP = {"bonafide": 0, "spoof": 1}
CLASS_NAMES = ["Real Audio", "Deepfake Audio"]
