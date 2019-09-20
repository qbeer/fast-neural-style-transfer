import tensorflow as tf
from .residual_block import ResidualBlock
from .convolutional_block import ConvolutionalBlock


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

    def call(self, x):
        x = self.conv1(x)
        #print('1 : ', x.get_shape().as_list())
        x = self.conv2(x)
        #print('2 : ', x.get_shape().as_list())
        x = self.conv3(x)
        #print(x.get_shape().as_list())
        x = self.res1(x)
        #print(x.get_shape().as_list())
        x = self.res2(x)
        #print(x.get_shape().as_list())
        x = self.res3(x)
        #print(x.get_shape().as_list())
        x = self.res4(x)
        #print(x.get_shape().as_list())
        x = self.res5(x)
        #print(x.get_shape().as_list())
        x = self.upconv1(x)
        #print(x.get_shape().as_list())
        x = self.upconv2(x)
        #print(x.get_shape().as_list())
        x = self.upconv3(x)
        #print(x.get_shape().as_list())
        return x
