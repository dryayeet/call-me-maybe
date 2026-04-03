# Importing required packages
import tensorflow as tf
import numpy as np
import argparse
import mediapipe as mp
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())


# Face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

vaModelPath = 'models/vaModel.tflite'
interpreter = tf.lite.Interpreter(model_path=vaModelPath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
vaTargetSize = tuple(input_details[0]['shape'][1:3])

cap = cv2.VideoCapture(0)

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output_va.avi", fourrcc, 22,
                                 (capWidth, capHeight))

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

    if not ret:
        break

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgbFrame)
    if results.detections:
        ih, iw, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * iw))
            y = max(0, int(bbox.ymin * ih))
            w = min(int(bbox.width * iw), iw - x)
            h = min(int(bbox.height * ih), ih - y)

            # Extract RGB face ROI from color frame
            face = frame[y:y + h, x:x + w]
            try:
                face = cv2.resize(face, vaTargetSize)
            except:
                continue

            # Preprocess: BGR to RGB, normalize to [-1, 1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype('float32')
            face = face / 255.0
            face = (face - 0.5) * 2.0
            face = np.expand_dims(face, 0)

            interpreter.set_tensor(input_details[0]['index'], face)
            interpreter.invoke()
            va_prediction = interpreter.get_tensor(output_details[0]['index'])
            valence = va_prediction[0][0]
            arousal = va_prediction[0][1]

            # Color based on valence: green (positive) to red (negative)
            val_norm = np.clip((valence + 1) / 2, 0, 1)  # map [-1,1] to [0,1]
            color = (int(50 + 150 * (1 - val_norm)),
                     int(50 + 150 * val_norm),
                     50)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20),
                     color, thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 160, y + h + 40),
                          color, -1)
            label = f"V:{valence:.2f} A:{arousal:.2f}"
            cv2.putText(frame, label,
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    if args["isVideoWriter"] == True:
        videoWrite.write(frame)

    cv2.imshow("Valence-Arousal Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
cv2.destroyAllWindows()
