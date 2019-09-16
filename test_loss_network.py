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
block2_conv2   (1, 128, 128, 128)
block1_conv2   (1, 256, 256, 64)
block3_conv3   (1, 64, 64, 256)
block4_conv3   (1, 32, 32, 512)
"""
for val in y.keys():
    print(val, " ", y[val].numpy().shape)
