import tensorflow as tf
import tensorflow_addons as tfa
from .reflection_pad_layer import ReflectionPadding2D


class ConvolutionalBlock(tf.keras.Model):
    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride=1,
                 upsample=False,
                 normalize=True,
                 relu=True):
        super(ConvolutionalBlock, self).__init__()
        self.upsample = upsample
        self.normalize = normalize
        self.relu = relu
        self.reflection_pad = ReflectionPadding2D(
            (kernel_size // 2, kernel_size // 2))
        self.conv = tf.keras.layers.Conv2D(out_channels,
                                           (kernel_size, kernel_size),
                                           strides=(stride, stride))
        self.upsample_layer = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')

        if self.normalize:
            self.instance_norm = tfa.layers.InstanceNormalization()

    def call(self, x, training=True):
        if self.upsample:
            x = self.upsample_layer(x)
        x = self.reflection_pad(x)
        x = self.conv(x)
        if self.normalize:
            x = self.instance_norm(x, training=training)
        if self.relu:
            x = tf.nn.relu(x)
        return x
