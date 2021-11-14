from flask import Flask, render_template, Response
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import os
import argparse
import numpy as np

app = Flask(__name__, static_folder='./static')
cap = cv2.VideoCapture(0)


def detect_mask(frame, face_detector, mask_detector, confidence_threshold):
    num_class = mask_detector.layers[-1].get_output_at(0).get_shape()[-1]
    labels, colors = None, None
    if num_class == 3:
        labels = ['Mask not incorrectly', 'Mask correctly', 'Please Wearing Mask']
        colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    print("[INFO] computing face detections...")
    face_detector.setInput(blob)
    detections = face_detector.forward()
    status = 0

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, w, h) = box.astype("int")
            (x, y) = (max(0, x), max(0, y))
            (w, h) = (min(w - 1, w), min(h - 1, h))
            face = frame[y:h, x:w]
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            prediction = mask_detector.predict(face)[0]
            label_idx = np.argmax(prediction)

            label = labels[label_idx]
            color = colors[label_idx]

            if num_class == 3:
                if label_idx == 0:
                    temp = 1
                elif label_idx == 1:
                    temp = 0
                else:
                    temp = 2
                status = max(status, temp)

            label = "{}: {:.2f}%".format(label, max(prediction) * 100)
            lines = 30
            thickness = 5
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            cv2.rectangle(frame, (x, y), (w, h), color, 1)
            cv2.line(frame, (x, y), (x + lines, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + lines), color, thickness)

            cv2.line(frame, (w, y), (w - lines, y), color, thickness)
            cv2.line(frame, (w, y), (w, y + lines), color, thickness)

            cv2.line(frame, (x, h), (x + lines, h), color, thickness)
            cv2.line(frame, (x, h), (x, h - lines), color, thickness)

            cv2.line(frame, (w, h), (w - lines, h), color, thickness)
            cv2.line(frame, (w, h), (w, h - lines), color, thickness)
        else:
            break

    return status, frame


parser = argparse.ArgumentParser()

parser.add_argument("--model", "-d",
                    default='ModelResult/data/datasets/model-pro-detect-mask5.h5',
                    help="dataset to train the model on")
parser.add_argument("--confidence", "-c", type=float, default=0.5,
                    help="minimum probability to filter weak face detections")
args = parser.parse_args()

current_full_dir = os.getcwd()
print("Current working directory: " + current_full_dir)
if current_full_dir.split("/")[-1] == "src":
    root = current_full_dir[:-4]
    os.chdir(root)
    print("Changed working directory to: " + root)

if args.confidence > 1 or args.confidence < 0:
    raise ValueError("Please provide a valid confidence value between 0 and 1 (inclusive).")

mask_detector_model_path = '../model-pro-detect-mask5.h5'
confidence_threshold = args.confidence
print("Mask detector save path: " + mask_detector_model_path)
print("Face detector thresholding confidence: " + str(confidence_threshold))
print("[INFO] loading face detector model...")
prototxt_path = "./face_detector/deploy.prototxt"
weights_path = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
face_detector = cv2.dnn.readNet(prototxt_path, weights_path)

print("[INFO] loading face mask detector model...")
mask_detector = load_model(mask_detector_model_path)

print("[INFO] starting video stream...")
capture = cv2.VideoCapture(0)


def generate_frames():
    while True:
        flags, frame = capture.read()
        detect_mask(frame, face_detector, mask_detector, confidence_threshold)
        ret1, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('test1.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
