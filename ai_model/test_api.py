# test_api.py

import requests
from pathlib import Path
# The URL where your Flask server is running
API_URL = "http://127.0.0.1:5000/predict"

# The path to an audio file you want to test
# Make sure this file exists in your folder!
SCRIPT_DIR = Path(__file__).resolve().parent
TEST_AUDIO_PATH = SCRIPT_DIR / "LA_T_1000137.flac"

def main():
    print(f"Sending {TEST_AUDIO_PATH} to the API...")
    
    # Open the audio file in binary read mode
    with open(TEST_AUDIO_PATH, 'rb') as audio_file:
        # Prepare the file for the POST request
        file_dict = {'file': (TEST_AUDIO_PATH.name, audio_file, 'audio/flac')}
        
        # Send the request to the server
        response = requests.post(API_URL, files=file_dict)
        
        # Print the result from the server
        if response.status_code == 200:
            print("✅ Success!")
            print(response.json())
        else:
            print("❌ Error!")
            print(f"Status Code: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    main()