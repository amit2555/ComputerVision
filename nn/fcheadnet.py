from keras.layers.core import Dropout, Dense, Flatten, Activation


class FCHeadNet(object):
    @staticmethod
    def build(baseModel, classes, neurons):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(neurons)(headModel)
        headModel = Activation("relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(classes)(headModel)
        headModel = Activation("softmax")(headModel)
        return headModel
