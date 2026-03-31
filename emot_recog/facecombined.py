# Importing required packages
import tensorflow as tf
import numpy as np
import argparse
import dlib
import cv2
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

emotions = {
    0: {"emotion": "Angry", "color": (193, 69, 42)},
    1: {"emotion": "Disgust", "color": (164, 175, 49)},
    2: {"emotion": "Fear", "color": (40, 52, 155)},
    3: {"emotion": "Happy", "color": (23, 164, 28)},
    4: {"emotion": "Sad", "color": (164, 93, 23)},
    5: {"emotion": "Suprise", "color": (218, 229, 97)},
    6: {"emotion": "Neutral", "color": (108, 72, 200)}
}


def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


# Face detection
faceLandmarks = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

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
    return np.argmax(pred), np.max(pred)


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


cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output_combined.avi", fourrcc, 22,
                                 (capWidth, capHeight))

executor = ThreadPoolExecutor(max_workers=2)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)

        grayFace = grayFrame[y:y + h, x:x + w]
        colorFace = frame[y:y + h, x:x + w]

        if grayFace.size == 0 or colorFace.size == 0:
            continue

        # Run both models in parallel
        emot_future = executor.submit(run_emotion, grayFace.copy())
        va_future = executor.submit(run_va, colorFace.copy())

        try:
            emotion_idx, emotion_prob = emot_future.result()
            valence, arousal = va_future.result()
        except:
            continue

        color = emotions[emotion_idx]['color']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Emotion label
        cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                 color, thickness=2)
        cv2.rectangle(frame, (x + 20, y + h + 20), (x + 160, y + h + 40),
                      color, -1)
        emot_label = emotions[emotion_idx]['emotion']
        if emotion_prob > 0.36:
            cv2.putText(frame, f"{emot_label} ({emotion_prob:.0%})",
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "---",
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # VA label
        cv2.rectangle(frame, (x + 20, y + h + 42), (x + 160, y + h + 62),
                      (60, 60, 60), -1)
        cv2.putText(frame, f"V:{valence:.2f} A:{arousal:.2f}",
                    (x + 25, y + h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Emotion + Valence-Arousal", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

executor.shutdown(wait=False)
cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
