import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import evidential_deep_learning as edl
from tensorflow.keras.models import Model


class Conv1DTranspose(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=2, padding="same"):
        """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
        """
        super(Conv1DTranspose, self).__init__()
        self.obj = keras.Sequential(
            [
                layers.Lambda(lambda x: tf.expand_dims(x, axis=2)),
                layers.Conv2DTranspose(
                    filters=filters,
                    kernel_size=(kernel_size, 1),
                    strides=(strides, 1),
                    padding=padding,
                ),
                layers.Lambda(lambda x: tf.squeeze(x, axis=2)),
            ]
        )

    def call(self, x):
        return self.obj(x)


class IncBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, size=15, strides=1):
        super(IncBlock, self).__init__()
        self.conv1x1 = layers.Conv1D(out_channels, kernel_size=1, use_bias=False)

        self.conv1 = keras.Sequential(
            [
                layers.Conv1D(
                    out_channels // 4, kernel_size=size, strides=strides, padding="same"
                ),
                layers.BatchNormalization(axis=1),
            ]
        )

        self.conv2 = keras.Sequential(
            [
                layers.Conv1D(out_channels // 4, kernel_size=1, use_bias=False),
                layers.BatchNormalization(axis=1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(
                    out_channels // 4,
                    kernel_size=size + 2,
                    strides=strides,
                    padding="same",
                ),
                layers.BatchNormalization(axis=1),
            ]
        )

        self.conv3 = keras.Sequential(
            [
                layers.Conv1D(out_channels // 4, kernel_size=1, use_bias=False),
                layers.BatchNormalization(axis=1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(
                    out_channels // 4,
                    kernel_size=size + 4,
                    strides=strides,
                    padding="same",
                ),
                layers.BatchNormalization(axis=1),
            ]
        )

        self.conv4 = keras.Sequential(
            [
                layers.Conv1D(out_channels // 4, kernel_size=1, use_bias=False),
                layers.BatchNormalization(axis=1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(
                    out_channels // 4,
                    kernel_size=size + 6,
                    strides=strides,
                    padding="same",
                ),
                layers.BatchNormalization(axis=1),
            ]
        )

        self.relu = layers.ReLU()

    def call(self, x):
        res = self.conv1x1(x)
        c1 = self.conv1(x)
        c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        concat = layers.concatenate([c1, c2, c3, c4], axis=-1)
        concat += res
        return self.relu(concat)





class Multi_class_CNN(tf.keras.Model):
    def __init__(self, in_channels):
        super(Multi_class_CNN, self).__init__()

        self.en1 = keras.Sequential(
            [
                layers.Conv1D(
                    32, kernel_size=3, padding="same", input_shape=in_channels
                ),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(32, kernel_size=5, strides=2, padding="same"),
                IncBlock(32, 32),
            ]
        )

        self.en2 = keras.Sequential(
            [
                layers.Conv1D(64, kernel_size=3, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(64, kernel_size=5, strides=2, padding="same"),
                IncBlock(64, 64),
            ]
        )

        self.en3 = keras.Sequential(
            [
                layers.Conv1D(128, kernel_size=3, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(128, kernel_size=3, strides=2, padding="same"),
                IncBlock(128, 128),
            ]
        )

        self.en4 = keras.Sequential(
            [
                layers.Conv1D(256, kernel_size=3, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(256, kernel_size=4, strides=2, padding="same"),
                IncBlock(256, 256),
            ]
        )

        self.en5 = keras.Sequential(
            [
                layers.Conv1D(512, kernel_size=3, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv1D(512, kernel_size=2, padding="same"),
                IncBlock(512, 512),
            ]
        )

        self.en6 = keras.Sequential(
            [
                layers.Conv1D(1024, kernel_size=3, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                IncBlock(1024, 1024),
            ]
        )

        self.en7_p = keras.Sequential(
            [
                layers.Conv1D(128, kernel_size=4, strides=2, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                IncBlock(128, 128),
            ]
        )

        self.en8_p = keras.Sequential(
            [
                layers.Conv1D(64, kernel_size=4, strides=2, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                IncBlock(64, 64),
            ]
        )

        self.en9_p = keras.Sequential(
            [
                layers.Conv1D(4, kernel_size=4, strides=2, padding="same"),
                layers.BatchNormalization(axis=-1),
                layers.LeakyReLU(alpha=0.2),
                IncBlock(4, 4),
            ]
        )

        self.fc = layers.Dense(1)
        # self.ev1 = edl.layers.DenseNormalGamma(1)

    def call(self, x):
        # import pdb;pdb.set_trace()
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)
        e4 = self.en4(e3)
        e5 = self.en5(e4)
        e6 = self.en6(e5)
        out_1 = self.en7_p(e6)
        out_2 = self.en8_p(out_1)
        out_3 = self.en9_p(out_2)
        out_4 = self.fc(tf.reshape(out_3, (-1, out_3.shape[1] * out_3.shape[2])))
        # out_5 = self.ev1(out_4)
        return tf.expand_dims(out_4, axis=1)

