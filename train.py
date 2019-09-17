# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.datasets.cifar10 import load_data

import matplotlib.pyplot as plt

from src import ModelTrainer

BUFFER_SIZE = 50000
BATCH_SIZE = 10
INPUT_SHAPE = 64

style_image = plt.imread('starry_night.jpg')
style_image = tf.Variable(style_image / 255., name='style_image')
style_image = [style_image] * BATCH_SIZE
style_image = tf.reshape(style_image,
                         shape=(BATCH_SIZE, *tf.shape(style_image[0]).numpy()))
style_image = tf.image.resize_images(style_image,
                                     size=(INPUT_SHAPE, INPUT_SHAPE))

trainer = ModelTrainer(style_image,
                       input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3),
                       n_classes=3,
                       batch_size=BATCH_SIZE)

(train, train_labels), (test, test_labels) = load_data()

# Resize
train = tf.reshape(train, shape=(train.shape[0], 32, 32, 3))
test = tf.reshape(test, shape=(test.shape[0], 32, 32, 3))

train = tf.image.resize_images(train, size=(INPUT_SHAPE, INPUT_SHAPE))
test = tf.image.resize_images(test, size=(INPUT_SHAPE, INPUT_SHAPE))

train = tf.data.Dataset.from_tensor_slices(
    train / 255.).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

trainer.train(train, lr=5e-3, epochs=5)