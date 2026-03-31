# Importing required packages
import tensorflow as tf
import numpy as np
import argparse
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())


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


faceLandmarks = "models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

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

    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)

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
