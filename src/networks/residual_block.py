import tensorflow as tf
import tensorflow_addons as tfa
from .convolutional_block import ConvolutionalBlock


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionalBlock(channels, 3, 1, False, True, True)
        self.conv2 = ConvolutionalBlock(channels, 3, 1, False, True, False)

        self.instance_norm1 = tfa.layers.InstanceNormalization()
        self.instance_norm2 = tfa.layers.InstanceNormalization()

    def __call__(self, x, training=True):
        _x = self.conv1(x)
        _x = self.instance_norm1(_x, training=training)
        _x = self.conv2(_x)
        _x = self.instance_norm2(_x, training=training)
        return x + _x
