# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
import pathlib
import os
import glob

import matplotlib.pyplot as plt

from src import ModelTrainer

BUFFER_SIZE = 50000
BATCH_SIZE = 4
WIDTH = 128
HEIGHT = 128

style_image = plt.imread('starry_night.jpg')
style_image = tf.Variable(style_image, name='style_image')
style_image = [style_image] * BATCH_SIZE
style_image = tf.reshape(style_image,
                         shape=(BATCH_SIZE, *tf.shape(style_image[0]).numpy()))
style_image = tf.image.resize_images(style_image, size=(WIDTH, HEIGHT))

trainer = ModelTrainer(style_image,
                       input_shape=(WIDTH, HEIGHT, 3),
                       n_classes=3,
                       batch_size=BATCH_SIZE)


def resize_image(img_path):
    img = tf.Variable(plt.imread(img_path))
    return tf.image.resize(img, [WIDTH, HEIGHT])


all_image_paths = list(glob.glob("/home/qbeer666/celeba_small/*"))
images = [resize_image(str(path)) for path in all_image_paths]

train = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE)

trainer.train(train, lr=5e-4, epochs=100)