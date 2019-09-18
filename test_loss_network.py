# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()

from src import LossNetwork

loss_net = LossNetwork()

x = tf.random.normal(shape=(1, 256, 256, 3))

y = loss_net(x)
"""
block5_conv2   (1, 16, 16, 512)
block1_conv1   (1, 256, 256, 64)
block2_conv1   (1, 128, 128, 128)
block3_conv1   (1, 64, 64, 256)
block4_conv1   (1, 32, 32, 512)
"""
for val in y.keys():
    print(val, " ", y[val].numpy().shape)
