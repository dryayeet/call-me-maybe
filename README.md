# Real-Time Emotion Recognition

Detects human faces in a live webcam feed and classifies facial expressions into one of seven emotions using a pre-trained deep learning model.

## Emotions Detected

`Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`

## Tech Stack

| Component | Tool |
|---|---|
| Face Detection | dlib (HOG-based frontal face detector) |
| Facial Landmarks | dlib 68-point shape predictor |
| Emotion Classification | Xception (transfer learning, trained on FER-2013) |
| Deep Learning Backend | TensorFlow / Keras |
| Image Processing | OpenCV |
| Language | Python |

## Project Structure

```
face-classification/
└── emotion-recognition/
    ├── emotionRecognition.py          # Main script
    └── models/
        ├── emotionModel.hdf5          # Pre-trained Xception emotion model
        └── dlib/
            └── shape_predictor_68_face_landmarks.dat
```

## How It Works

1. Captures frames from webcam via OpenCV
2. Converts frame to grayscale and runs dlib's frontal face detector
3. Extracts face ROI, resizes to 64x64, normalizes pixel values to [-1, 1]
4. Feeds processed face into the Xception-based classifier
5. Overlays predicted emotion label on frame if confidence > 0.36

## Setup

```bash
pip install tensorflow keras opencv-python dlib-bin numpy
```

## Usage

```bash
cd face-classification/emotion-recognition
python emotionRecognition.py
```

Optional — save output to video:

```bash
python emotionRecognition.py -vw True
```

Press `Esc` to exit.

## Model Details

- **Architecture:** Xception (mini variant)
- **Dataset:** FER-2013 (35,887 48x48 grayscale images)
- **Input:** 64x64 single-channel grayscale
- **Output:** 7-class softmax probability distribution
- **Confidence threshold:** 0.36

## License

See [LICENSE](face-classification/LICENSE).
