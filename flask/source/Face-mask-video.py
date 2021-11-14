from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, w, h) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (x, y) = (max(0, x), max(0, y))
            (w, h) = (min(w - 1, w), min(h - 1, h))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[y:h, x:w]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((x, y, w, h))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="model-mask3.pro",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()
    if not ret:
        break
    else:
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        lines = 30
        thickness = 5
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (x, y, w, h) = box
            (mask, withoutMask) = pred
            x1 = x + w
            y1 = y + h

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Wearing Mask" if mask > withoutMask else "No Wearing Mask"
            color = (0, 255, 0) if label == "Wearing Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (x - 20, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
            cv2.rectangle(frame, (x, y), (w, h), color, 1)
            cv2.line(frame, (x, y), (x + lines, y), color, thickness)
            cv2.line(frame, (x, y), (x, y + lines), color, thickness)

            cv2.line(frame, (w, y), (w - lines, y), color, thickness)
            cv2.line(frame, (w, y), (w, y + lines), color, thickness)

            cv2.line(frame, (x, h), (x + lines, h), color, thickness)
            cv2.line(frame, (x, h), (x, h - lines), color, thickness)

            cv2.line(frame, (w, h), (w - lines, h), color, thickness)
            cv2.line(frame, (w, h), (w, h - lines), color, thickness)
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
