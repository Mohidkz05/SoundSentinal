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

# --- LFCC front-end ---------------------------------------------------------
# 20 static coefficients plus delta and delta-delta = 60 channels. That is the
# configuration the official ASVspoof2019 LFCC-GMM baseline uses, chosen so our
# LFCC row is comparable to theirs rather than to a variant of our own
# invention.
N_LFCC = 20
N_LFCC_FILTER = 20

# The front-end is a property of a trained model, not a global setting: a
# checkpoint trained on LFCC is meaningless if served log-Mel. Every checkpoint
# records which one it used, and app.py and evaluate.py read it back. "logmel"
# is the default so every existing checkpoint, which predates this, keeps
# working.
FRONTENDS = ("logmel", "lfcc")
DEFAULT_FRONTEND = "logmel"


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


class _WithDeltas(nn.Module):
    """Stack a feature map with its first and second time derivatives.

    Cepstral features describe one frame in isolation; the deltas are how the
    spectrum is *changing*, and synthesis artefacts often live in that motion
    rather than in any single frame. The ASVspoof baselines use them, so we do
    too — otherwise our LFCC row would not be measuring the same thing theirs is.
    """

    def __init__(self):
        super().__init__()
        self.deltas = T.ComputeDeltas()

    def forward(self, x):
        d1 = self.deltas(x)
        d2 = self.deltas(d1)
        return torch.cat((x, d1, d2), dim=-2)


def build_transform(frontend=DEFAULT_FRONTEND):
    """Waveform -> time-frequency features. Use this everywhere audio becomes a tensor.

    `logmel` is the original pipeline. `lfcc` exists because the Mel scale is
    designed to mimic human hearing and therefore compresses high frequencies —
    which is exactly where vocoder and waveform-filtering artefacts live. Both
    official ASVspoof2019 baselines are cepstral for that reason. Swapping only
    this, with the model and schedule held fixed, is a controlled test of
    whether the front-end is what limits us. See "Per-attack, against the
    official baselines" in APPROACH.md for the A17 result that motivates it.
    """
    if frontend not in FRONTENDS:
        raise ValueError(f"frontend must be one of {FRONTENDS}, got {frontend!r}")

    if frontend == "lfcc":
        return nn.Sequential(
            T.LFCC(
                sample_rate=SAMPLE_RATE,
                n_filter=N_LFCC_FILTER,
                n_lfcc=N_LFCC,
                speckwargs={"n_fft": N_FFT, "hop_length": HOP_LENGTH},
            ),
            _WithDeltas(),
        )

    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        ),
        T.AmplitudeToDB(stype="power", top_db=TOP_DB),
    )


def preprocess_waveform(waveform, sample_rate, transform_pipeline, max_len=MAX_LEN):
    """
    Waveform -> standardized features, shape (1, C, frames).

    C is 128 for the log-Mel front-end and 60 for LFCC (20 coefficients plus
    two delta orders). The network tolerates both because it pools adaptively.

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
