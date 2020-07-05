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

import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()


def train(args):
    BATCH_SIZE = args.batch_size
    WIDTH = args.width
    HEIGHT = args.height
    STYLE = './style_images/colorful_portrait.jpg' if not args.starry_night else './style_images/starry_night.png'

    style_image = plt.imread(STYLE)
    style_image = tf.keras.applications.vgg16.preprocess_input(style_image)
    style_image = tf.constant(style_image, name='style_image')
    style_image = tf.expand_dims(style_image, 0)
    style_image = tf.image.resize(style_image, size=(HEIGHT, WIDTH))

    trainer = ModelTrainer(style_image,
                           residual=True,
                           input_shape=(HEIGHT, WIDTH, 3),
                           c_out=3,
                           batch_size=BATCH_SIZE)

    dataset = tfds.load('imagenette', split='train', batch_size=BATCH_SIZE)

    def preproc_image_mapper(example):
        image = tf.cast(example['image'], tf.float32)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        return tf.image.resize(image, size=(HEIGHT, WIDTH))

    dataset = dataset.map(preproc_image_mapper).cache()

    trainer.train(dataset, lr=1e-3, epochs=args.epochs)

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--epochs', type=int, default=2, required=False)
parser.add_argument('--width', type=int, default=320, required=False)
parser.add_argument('--height', type=int, default=2, required=False)
parser.add_argument('--starry_night', default=False, action='store_true', required=False)
