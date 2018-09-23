from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import argparse
import numpy as np
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascade", help="path to cascade xml", required=True)
parser.add_argument("-m", "--model", help="path to pre-trained model", required=True)
args = vars(parser.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in rects:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        (not_smiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > not_smiling else "Not Smiling"
        cv2.putText(frame_clone, label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face", frame_clone)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

