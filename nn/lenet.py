from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras import backend as K


class LeNet(object):
    @staticmethod
    def build(width, height, depth, classes):
        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width)
        else:
            input_shape = (height, width, depth)

        model = Sequential()

        # INPUT => CONV2D => RELU/TANH => POOL
        model.add(Conv2D(20, (5, 5), input_shape=input_shape, padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # CONV2D => RELU/TANH => POOL
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # FC => RELU/TANH => FC
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
