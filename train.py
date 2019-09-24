# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
tf.enable_eager_execution()
import pathlib
import glob
import numpy as np

import matplotlib.pyplot as plt

from multiprocessing import Pool

from src import ModelTrainer

BUFFER_SIZE = 5000
BATCH_SIZE = 4
WIDTH = 256
HEIGHT = 168

style_image = plt.imread('./style_images/van_gogh.jpg')
style_image = tf.Variable(style_image, name='style_image')
style_image = tf.expand_dims(style_image, 0)
style_image = tf.image.resize_images(style_image, size=(WIDTH, HEIGHT))

trainer = ModelTrainer(style_image,
                       residual=True,
                       input_shape=(WIDTH, HEIGHT, 3),
                       n_classes=3,
                       batch_size=BATCH_SIZE)

raw_image_dataset = tf.data.TFRecordDataset('celeba.tfrecords')

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)


def _resize_image_function(example_parsed):
    image = tf.image.decode_jpeg(example_parsed['image'])
    image = tf.cast(image, tf.float32)
    height = tf.cast(example_parsed['height'], tf.int32)
    width = tf.cast(example_parsed['width'], tf.int32)

    image = tf.reshape(image, [height, width, 3])
    return tf.image.resize(image, size=(WIDTH, HEIGHT))


parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
resized_image_dataset = parsed_image_dataset.map(_resize_image_function)

train = resized_image_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

trainer.train(train, lr=1e-3, epochs=2)