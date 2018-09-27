from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from nn.fcheadnet import FCHeadNet
from nn.preprocessing.simpledatasetloader import SimpleDatasetLoader
from nn.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from nn.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from imutils import paths
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-m", "--model", help="path to output model", required=True)
args = vars(parser.parse_args())

# Data Augmentation 
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

image_paths = list(paths.list_images(args["dataset"]))
class_names = [p.split(os.path.sep)[-2] for p in image_paths]
class_names = [str(x) for x in np.unique(class_names)]


# Load data and preprocess
iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(224, 224)
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


# Load model
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = FCHeadNet.build(baseModel, len(class_names), 256)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

optimizer = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.fit_generator(aug.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test), epochs=20,
                    steps_per_epoch=len(X_train)//32, verbose=1)

preds = model.predict(X_test, batch_size=32)
print classification_report(preds.argmax(axis=1), y_test.argmax(axis=1), target_names=class_names)

model.save(args["model"])


