# app.py

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from pathlib import Path
# ===================================================================
# 1. RE-DEFINE THE EXACT SAME MODEL ARCHITECTURE
# This must be identical to the class in your training script.
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
        # Note: Dropout is automatically disabled in .eval() mode
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ===================================================================
# 2. LOAD THE TRAINED MODEL FROM THE SAVED FILE
# ===================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "deepfake_audio_detector.pth"
model = AudioClassifierCNN()

# Load the weights from the file into the model structure
# map_location='cpu' ensures it runs on any computer, even without a GPU
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# Set the model to evaluation mode (very important!)
# This turns off training-specific features like dropout.
model.eval()

class_names = ["Real Audio", "Deepfake Audio"]

# ===================================================================
# 3. DEFINE THE PREPROCESSING FOR A SINGLE AUDIO FILE
# This must apply the *exact same* transformations as the training data.
# ===================================================================
def preprocess_audio(audio_file, max_len=64000):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Resample if necessary to match the training sample rate
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Pad or truncate to the same fixed length used in training
    if waveform.shape[1] > max_len:
        waveform = waveform[:, :max_len]
    else:
        padding = max_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    spectrogram = transform(waveform)
    # Add a "batch" dimension for the model
    return spectrogram.unsqueeze(0) 

# ===================================================================
# 4. CREATE THE FLASK WEB SERVER
# ===================================================================
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['file']
    
    try:
        # 1. Preprocess the incoming audio file
        tensor = preprocess_audio(audio_file)
        
        # 2. Make a prediction (no gradients needed)
        with torch.no_grad():
            outputs = model(tensor)
            # Get the predicted class index (0 or 1)
            _, predicted_idx = torch.max(outputs.data, 1)
            predicted_label = class_names[predicted_idx.item()]
            
            # 3. Calculate the confidence score
            probabilities = F.softmax(outputs, dim=1)
            confidence = probabilities[0][predicted_idx.item()].item()

        # 4. Send the result back as JSON
        return jsonify({
            "prediction": predicted_label,
            "confidence": f"{confidence:.2%}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Start the server
    print("Starting Flask server... Visit http://127.0.0.1:5000")
    app.run(debug=True)