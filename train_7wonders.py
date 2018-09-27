from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from nn.fcheadnet import FCHeadNet
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="save model in directory", required=True)
parser.add_argument("-d", "--dataset", help="path to dataset", required=True)
args = vars(parser.parse_args())

data_augment = {"rotation_range": 30,
                "rescale": 1.0/255,
                "shear_range": 0.2,
                "zoom_range": 0.2,
                "horizontal_flip": True}

train_datagen = ImageDataGenerator(**data_augment)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.sep.join([args["dataset"], "train"]),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42)

valid_generator = valid_datagen.flow_from_directory(
    directory=os.path.sep.join([args["dataset"], "valid"]),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42)


baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = FCHeadNet.build(baseModel, len(train_generator.class_indices), 256)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

optimizer = SGD(lr=0.001)
model.compile(metrics=["accuracy"], optimizer=optimizer, loss="categorical_crossentropy")
H = model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n // train_generator.batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n // valid_generator.batch_size,
                        epochs=20,
                        verbose=1)

model.save(args["model"])                                      # save model to disk

