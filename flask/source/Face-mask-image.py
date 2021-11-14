# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.xception import preprocess_input
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
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

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
	default="./ModelResult/data/datasets/model-pro-detect-mask1.h5",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
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
print("[INFO] starting caculate image...")
frame = cv2.imread('./data/datasets/incorrect_mask/aug_1.jpg')
scale_percent = 40  # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# loop over the frames from the video stream


(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
		# locations
lines = 30
thickness = 5
for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
	(x, y, w, h) = box
	(mask, withoutMask,incorrect_mask) = pred
	x1 = x+w
	y1 = y+h

			# determine the class label and color we'll use to draw
			# the bounding box and text
	label = "Wearing Mask" if mask > withoutMask and mask > incorrect_mask else "No Wearing Mask" or "incorrect_mask"
	color = (0,255, 0) if label == "Wearing Mask" else (0, 0, 255)

			# include the probability in the label
	label = "{}: {:.2f}%".format(label, max(mask, withoutMask,incorrect_mask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
	cv2.putText(frame, label, (x- 20, y - 10),
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
cv2.imwrite('output'+'/'+'bia'+'.jpg', frame)
cv2.waitKey(0)

cv2.destroyAllWindows()

