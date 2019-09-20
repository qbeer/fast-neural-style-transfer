import tensorflow as tf
from .convolutional_block import ConvolutionalBlock


class ResidualBlock(tf.keras.Model):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvolutionalBlock(channels, 3, 1, False, True, True)
        self.conv2 = ConvolutionalBlock(channels, 3, 1, False, True, False)

    def call(self, x):
        _x = self.conv1(x)
        _x = self.conv2(x)
        return tf.keras.layers.Add()([_x, x])
