# Importing required packages
import tensorflow as tf
import numpy as np
import argparse
import mediapipe as mp
import cv2
from concurrent.futures import ThreadPoolExecutor
from affect_fusion import AffectFusionEngine

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

# Base emotion classes (direct model output, always displayed)
BASE_EMOTIONS = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

# Face detection
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
face_detection = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path='models/blaze_face_short_range.tflite'),
        min_detection_confidence=0.5
    )
)

# Emotion model (TFLite)
emotInterp = tf.lite.Interpreter(model_path='models/emotionModel.tflite')
emotInterp.allocate_tensors()
emotInput = emotInterp.get_input_details()
emotOutput = emotInterp.get_output_details()
emotTargetSize = tuple(emotInput[0]['shape'][1:3])

# VA model (TFLite)
vaInterp = tf.lite.Interpreter(model_path='models/vaModel.tflite')
vaInterp.allocate_tensors()
vaInput = vaInterp.get_input_details()
vaOutput = vaInterp.get_output_details()
vaTargetSize = tuple(vaInput[0]['shape'][1:3])

# Fusion engine
engine = AffectFusionEngine()


def run_emotion(grayFace):
    face = cv2.resize(grayFace, emotTargetSize)
    face = face.astype('float32')
    face = face / 255.0
    face = (face - 0.5) * 2.0
    face = np.expand_dims(face, 0)
    face = np.expand_dims(face, -1)
    emotInterp.set_tensor(emotInput[0]['index'], face)
    emotInterp.invoke()
    pred = emotInterp.get_tensor(emotOutput[0]['index'])
    return np.argmax(pred), np.max(pred), pred[0]


def run_va(colorFace):
    face = cv2.resize(colorFace, vaTargetSize)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype('float32')
    face = face / 255.0
    face = (face - 0.5) * 2.0
    face = np.expand_dims(face, 0)
    vaInterp.set_tensor(vaInput[0]['index'], face)
    vaInterp.invoke()
    pred = vaInterp.get_tensor(vaOutput[0]['index'])
    return pred[0][0], pred[0][1]


def draw_va_hud(frame, result):
    size = 150
    pad = 10
    hx = frame.shape[1] - size - pad  # top-right corner
    hy = pad
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (hx, hy), (hx + size, hy + size + 25), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "Valence-Arousal", (hx + 18, hy + 14),
                font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Grid area below title
    gy = hy + 20
    gs = size
    cx, cy = hx + gs // 2, gy + gs // 2

    # Axes
    cv2.line(frame, (hx + 2, cy), (hx + gs - 2, cy), (70, 70, 70), 1)
    cv2.line(frame, (cx, gy + 2), (cx, gy + gs - 2), (70, 70, 70), 1)

    # Quadrant labels
    fs, fc = 0.25, (100, 100, 100)
    cv2.putText(frame, "Pleasant+", (cx + 4, gy + 12), font, fs, fc, 1, cv2.LINE_AA)
    cv2.putText(frame, "Unpleasant+", (hx + 2, gy + 12), font, fs, fc, 1, cv2.LINE_AA)
    cv2.putText(frame, "Pleasant-", (cx + 4, gy + gs - 5), font, fs, fc, 1, cv2.LINE_AA)
    cv2.putText(frame, "Unpleasant-", (hx + 2, gy + gs - 5), font, fs, fc, 1, cv2.LINE_AA)

    # VA dot
    v, a = result.valence_smooth, result.arousal_smooth
    dx = int(cx + v * (gs // 2))
    dy = int(cy - a * (gs // 2))
    dx = int(np.clip(dx, hx + 3, hx + gs - 3))
    dy = int(np.clip(dy, gy + 3, gy + gs - 3))
    cv2.circle(frame, (dx, dy), 5, result.plutchik_color, -1)
    cv2.circle(frame, (dx, dy), 5, (255, 255, 255), 1)

    # Numeric readout below grid
    txt = f"V:{v:+.2f}  A:{a:+.2f}"
    cv2.putText(frame, txt, (hx + 18, gy + gs + 18),
                font, 0.35, (180, 180, 180), 1, cv2.LINE_AA)


cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output_combined.avi", fourrcc, 22,
                                 (capWidth, capHeight))

executor = ThreadPoolExecutor(max_workers=2)
last_result = None

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
    results = face_detection.detect(mp_image)

    if not results.detections:
        engine.reset()
    else:
        for detection in results.detections:
            bbox = detection.bounding_box
            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            w = bbox.width
            h = bbox.height

            grayFace = grayFrame[y:y + h, x:x + w]
            colorFace = frame[y:y + h, x:x + w]

            if grayFace.size == 0 or colorFace.size == 0:
                continue

            # Run both models in parallel
            emot_future = executor.submit(run_emotion, grayFace.copy())
            va_future = executor.submit(run_va, colorFace.copy())

            try:
                emotion_idx, emotion_prob, softmax_probs = emot_future.result()
                valence, arousal = va_future.result()
            except:
                continue

            # Fuse results
            result = engine.update(emotion_idx, emotion_prob, softmax_probs,
                                   valence, arousal)
            last_result = result

            # Bounding box (dimmed if unreliable)
            color = result.plutchik_color
            if not result.is_reliable:
                color = tuple(c // 2 for c in color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # --- Face labels (directly below bbox) ---
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Line 1: Base emotion + confidence (always shown)
            base_name = BASE_EMOTIONS[result.emotion_idx]
            reliability = "*" if not result.is_reliable else ""
            line1 = f"{base_name} {result.emotion_prob:.0%}{reliability}"
            tw1 = cv2.getTextSize(line1, font, 0.45, 1)[0][0]
            box_w = tw1 + 14
            label_y = y + h + 2
            cv2.rectangle(frame, (x, label_y), (x + box_w, label_y + 20),
                          color, -1)
            cv2.putText(frame, line1, (x + 5, label_y + 15),
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Line 2: Plutchik intensity word (only if adds info)
            next_y = label_y + 22
            if result.plutchik_petal and result.plutchik_label != base_name:
                line2 = f"~ {result.plutchik_label}"
                tw2 = cv2.getTextSize(line2, font, 0.38, 1)[0][0]
                cv2.rectangle(frame, (x, next_y), (x + tw2 + 14, next_y + 18),
                              (50, 50, 50), -1)
                cv2.putText(frame, line2, (x + 5, next_y + 14),
                            font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
                next_y += 20

            # Line 3: Dyad (only if detected)
            if result.dyad_name:
                line3 = f"+ {result.dyad_name}"
                tw3 = cv2.getTextSize(line3, font, 0.38, 1)[0][0]
                cv2.rectangle(frame, (x, next_y), (x + tw3 + 14, next_y + 18),
                              (60, 45, 30), -1)
                cv2.putText(frame, line3, (x + 5, next_y + 14),
                            font, 0.38, (200, 220, 255), 1, cv2.LINE_AA)

    # Draw VA HUD (top-right, last processed face)
    if last_result is not None:
        draw_va_hud(frame, last_result)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Real-Time Affect Analysis", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

executor.shutdown(wait=False)
cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
