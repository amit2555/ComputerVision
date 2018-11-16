from nn.minigooglenet import MiniGoogLeNet
from callbacks.custom_callbacks import LogLearningRate
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os


BATCH_SIZE = 64
MAX_EPOCHS = 70
INITIAL_LR = 1e-2
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def polynomial_decay(epoch):
    """Equation for polynomial decay is a = a0 * [(1-(epoch/max_epochs)) ** p]
       a = new learning-rate
       a0 = initial learning-rate
       epoch = current epoch
       max_epochs = total number of epochs
       p = polynomial"""

    p = 2.0
    return INITIAL_LR * (1-(epoch/float(MAX_EPOCHS)))**p


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, help="path to save model")
parser.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(parser.parse_args())

# Normalize
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

# Data Augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Changing learning rate based on a polynomial decay wrt current epoch
lr_log = LogLearningRate(args["output"])
callbacks = [LearningRateScheduler(polynomial_decay), lr_log]

optimizer = SGD(lr=INITIAL_LR, momentum=0.9)
model = MiniGoogLeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy",
              metrics=["accuracy"])
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_test, y_test),
                        steps_per_epoch=len(X_train)//BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        callbacks=callbacks,
                        verbose=1)

model.save(args["model"])

preds = model.predict(X_test)
print classification_report(y_test.argmax(axis=1), preds.argmax(axis=1), target_names=CLASSES)
