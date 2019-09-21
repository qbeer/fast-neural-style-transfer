import tensorflow as tf
from .residual_block import ResidualBlock
from .convolutional_block import ConvolutionalBlock
from .output_scale_layer import OutputScale


class ResidualInferenceNetwork(tf.keras.Model):
    def __init__(self):
        super(ResidualInferenceNetwork, self).__init__()
        self.conv1 = ConvolutionalBlock(32, kernel_size=9, stride=1)
        self.conv2 = ConvolutionalBlock(64, kernel_size=3, stride=2)
        self.conv3 = ConvolutionalBlock(128, kernel_size=3, stride=2)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.upconv1 = ConvolutionalBlock(64, kernel_size=3, upsample=True)
        self.upconv2 = ConvolutionalBlock(32, kernel_size=3, upsample=True)
        self.upconv3 = ConvolutionalBlock(3,
                                          kernel_size=9,
                                          normalize=False,
                                          relu=False)
        self.output_scale = OutputScale()

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = tf.nn.tanh(x)
        x = self.output_scale(x)
        return x
