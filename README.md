# Real-Time Facial Affect Analysis

Detects human faces in a live webcam feed and performs two types of affect analysis in parallel:
- **Categorical:** Classifies expressions into 7 discrete emotions
- **Dimensional:** Predicts continuous Valence (positive/negative) and Arousal (calm/excited) values

## Tech Stack

| Component | Tool |
|---|---|
| Face Detection | dlib (HOG-based frontal face detector) |
| Facial Landmarks | dlib 68-point shape predictor |
| Emotion Classification | Mini Xception (trained on FER-2013) |
| Valence-Arousal Regression | MobileNetV2 (trained on AffectNet) |
| Inference Runtime | TensorFlow Lite |
| Parallel Execution | Python `concurrent.futures.ThreadPoolExecutor` |
| Image Processing | OpenCV |

## Project Structure

```
emot_recog/
├── facecombined.py              # Combined emotion + VA (parallel inference)
├── facemot.py                   # Standalone emotion classification
├── faceva.py                    # Standalone valence-arousal regression
├── convert_to_tflite.py         # HDF5 → TFLite (emotion model)
├── convert_va_to_tflite.py      # H5 weights → TFLite (VA model)
└── models/
    ├── emotionModel.tflite      # Emotion classifier (236 KB)
    ├── vaModel.tflite           # VA regressor (13.8 MB)
    ├── emotionModel.hdf5        # Original emotion weights
    ├── regressor_weights.h5     # Original VA weights
    └── dlib/
        └── shape_predictor_68_face_landmarks.dat
```

## How It Works

1. Captures frames from webcam via OpenCV
2. Converts to grayscale → dlib face detection → extracts face ROI per detection
3. Two models run **in parallel** per face via thread pool:
   - **Emotion:** Grayscale ROI → 64x64 → normalize to [-1, 1] → Mini Xception → 7-class softmax
   - **VA:** Color ROI → 224x224 RGB → normalize to [-1, 1] → MobileNetV2 → [valence, arousal]
4. Overlays emotion label (with confidence) and V/A values on the frame

## Setup

```bash
pip install tensorflow opencv-python dlib-bin numpy
```

## Usage

**Combined (recommended):**
```bash
cd emot_recog
python facecombined.py
```

**Emotion only:**
```bash
python facemot.py
```

**Valence-Arousal only:**
```bash
python faceva.py
```

All scripts support `-vw True` to save output to video. Press `Esc` to exit.

## Models

| | Emotion | Valence-Arousal |
|---|---|---|
| **Architecture** | Mini Xception | MobileNetV2 + Dense(1024) + Dense(2) |
| **Dataset** | FER-2013 (35,887 images) | AffectNet (400K+ images) |
| **Input** | 64x64 grayscale | 224x224 RGB |
| **Output** | 7-class softmax | 2 floats: valence, arousal |
| **Params** | 58K | 3.6M |
| **TFLite size** | 236 KB | 13.8 MB |

### Emotion Classes

`Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`

### Valence-Arousal Scale

- **Valence** [-1, 1]: Negative ← 0 → Positive
- **Arousal** [-1, 1]: Calm ← 0 → Excited

## License

See [LICENSE](face-classification/LICENSE).
