# FaceGuard - AI-Based Secure Login System

## Features
- Face recognition login (replaces passwords)
- Emotion detection (only allows calm/neutral users)
- Offline processing (no external APIs)
- Web interface with live camera feed

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open browser to `http://localhost:5000`

## Usage

1. **Register**: Enter your name and click "Register Face"
2. **Login**: Click "Login" - system checks face + emotion
3. **Security**: Only calm/neutral emotions grant access

## How it Works

- Uses `face_recognition` library for face encoding/matching
- `FER` library detects emotions in real-time
- Flask backend processes images from webcam
- All processing happens locally (offline)

## Files Structure
- `app.py` - Flask backend with AI logic
- `templates/index.html` - Frontend with webcam
- `faces/` - Stored face images (auto-created)
