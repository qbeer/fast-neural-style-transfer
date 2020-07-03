# Tensorflow throws a bunch of `FutureWarning`s
import glob
import pathlib
import warnings
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from src import ModelTrainer

warnings.simplefilter(action='ignore', category=FutureWarning)

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()

BUFFER_SIZE = 5000
BATCH_SIZE = 4
WIDTH = 160
HEIGHT = 320

style_image = plt.imread('./style_images/colorful_portrait.jpg')
style_image = tf.keras.applications.vgg16.preprocess_input(style_image)
style_image = tf.constant(style_image, name='style_image')
style_image = tf.expand_dims(style_image, 0)
style_image = tf.image.resize(style_image, size=(WIDTH, HEIGHT))

trainer = ModelTrainer(style_image,
                       residual=True,
                       input_shape=(WIDTH, HEIGHT, 3),
                       c_out=3,
                       batch_size=BATCH_SIZE)

dataset = tfds.load('imagenette', split='train', batch_size=BATCH_SIZE)


def preproc_image_mapper(example):
    image = tf.cast(example['image'], tf.float32)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return tf.image.resize(image, size=(WIDTH, HEIGHT))


dataset = dataset.map(preproc_image_mapper).cache()

trainer.train(dataset, lr=1e-3, epochs=1)
