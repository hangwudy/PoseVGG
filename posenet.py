# import the necessary packages
from keras import applications
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf


class PoseNet:

    @staticmethod
    def ResNet_mod(width, height, numLatitudes, numLongitudes,
                   finalAct="softmax"):
        base_model = applications.ResNet50(weights="imagenet", include_top=False, input_shape=(width, height, 3))
        x = base_model.output

        x = GlobalAveragePooling2D()(x)
        # x_la = Dense(128, activation='relu', name='fc1_1')(x)
        # x_la = Dense(1024, activation='relu', name='fc2_1')(x_la)
        x_la = Dense(numLatitudes)(x)
        latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

        # x_lo = Dense(128, activation='relu', name='fc1_2')(x)
        # x_lo = Dense(1024, activation='relu', name='fc2_2')(x_lo)
        x_lo = Dense(numLongitudes)(x)
        longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

        model = Model(
            inputs=base_model.input,
            outputs=[latitudeBranch, longitudeBranch],
            name='posenet')

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        for layer in model.layers[:30]:
            layer.trainable = False

        return model

    @staticmethod
    def VGG16_mod(width, height, numLatitudes, numLongitudes,
                  finalAct="softmax"):
        base_model = applications.VGG16(weights="imagenet", include_top=False, input_shape=(width, height, 3))
        x = base_model.output

        x = Flatten()(x)

        x_la = Dense(1024, activation='relu', name='fc1_1')(x)
        x_la = Dense(1024, activation='relu', name='fc2_1')(x_la)
        x_la = Dense(numLatitudes)(x_la)
        latitudeBranch = Activation(finalAct, name="latitude_output")(x_la)

        x_lo = Dense(1024, activation='relu', name='fc1_2')(x)
        x_lo = Dense(1024, activation='relu', name='fc2_2')(x_lo)
        x_lo = Dense(numLongitudes)(x_lo)
        longitudeBranch = Activation(finalAct, name="longitude_output")(x_lo)

        model = Model(
            inputs=base_model.input,
            outputs=[latitudeBranch, longitudeBranch],
            name='posenet')

        for i, layer in enumerate(model.layers):
            print(i, layer.name)

        for layer in model.layers[:10]:
            layer.trainable = False

        return model
