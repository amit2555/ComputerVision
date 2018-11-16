from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input, concatenate
from keras.models import Model
from keras import backend as K


class MiniGoogLeNet(object):
    @staticmethod
    def conv_module(x, filters, w, h, stride, chanDim, padding="same"):
        """ filters: Number of filters
            w      : filter width
            h      : filter height
            stride : stride size
            chanDim: channel dimension - channel first or channel last
            padding: type of padding"""

        x = Conv2D(filters, (w, h), strides=stride, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, numK1x1, numK3x3, chanDim):
        conv_1x1 = MiniGoogLeNet.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
        x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
        return x

    @staticmethod
    def downsample_module(x, numK3x3, chanDim):
        conv_3x3 = MiniGoogLeNet.conv_module(x, numK3x3, 3, 3, (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = concatenate([conv_3x3, pool], axis=chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1

        inputs = Input(shape=input_shape)

        # first a Convolution module
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        # 2xInception modules followed by a Downsample module
        x = MiniGoogLeNet.inception_module(x, 32, 32, chanDim) # total filters after merge : 32+32=64
        x = MiniGoogLeNet.inception_module(x, 32, 48, chanDim) # total filters after merge : 32+48=80
        x = MiniGoogLeNet.downsample_module(x, 80, chanDim) # total 3x3 filters 80

        # 4xInception modules followed by a Downsample module
        x = MiniGoogLeNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogLeNet.downsample_module(x, 96, chanDim)

        # 2xInception modules followed by an Average pooling
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(0.5)(x)

        # Fully-connected layer
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="googlenet")
        return model

