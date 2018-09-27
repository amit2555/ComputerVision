from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
from nn.lenet import LeNet
import matplotlib.pyplot as plt
import argparse
import numpy as np
import imutils
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="save model in directory", required=True)
parser.add_argument("-d", "--dataset", help="path to dataset", required=True)
args = vars(parser.parse_args())


data = []
labels = []

# Load images and preprocess
for image_path in list(paths.list_images(args["dataset"])):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            # convert to grayscale
    image = imutils.resize(image, height=28, width=28)         # resize to 28x28
    image = img_to_array(image)                                # convert to array
    data.append(image)

    label = image_path.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255                     # normalize pixels to range [0, 1]
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)      # one-hot encoding

class_total = labels.sum(axis=1)
class_weight = class_total.max() / class_total

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels,
                                                    random_state=42)
# This stratify parameter makes a split so that the proportion of values in the sample produced will be the same as the proportion of values provided to parameter stratify

epochs = 20
model = LeNet.build(width=28, height=28, depth=1, classes=2)
optimizer = SGD(lr=0.01, decay=0.01/epochs, momentum=0.9, nesterov=True)
model.compile(metrics=["accuracy"], optimizer="adam", loss=["binary_crossentropy"])
H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              class_weight=class_weight, batch_size=32, epochs=epochs, verbose=1)

preds = model.predict(X_test)
print classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=le.classes_)

model.save(args["model"])                                      # save model to disk

