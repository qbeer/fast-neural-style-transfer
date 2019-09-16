# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.datasets.cifar10 import load_data

import matplotlib.pyplot as plt

from src import ModelTrainer

style_image = plt.imread('van_gogh.png')
style_image = tf.Variable(style_image, name='style_image')
style_image = tf.reshape(shape=(1, *tf.shape(style_image).numpy()))
style_image = tf.image.resize_images(style_image, size=(64, 64))

print(style_image.numpy().shape)

trainer = ModelTrainer(style_image, input_shape=(64, 64, 3), n_classes=3)