# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from datetime import datetime

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow import keras

from src import ResidualInferenceNetwork

IMAGE_SIZE = 128

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

(train, train_labels), (test, test_labels) = load_data()

# Resize
train = tf.reshape(train[:800] / 255., shape=(800, 32, 32, 3))
test = tf.reshape(test[:16] / 255., shape=(16, 32, 32, 3))

train = tf.image.resize_images(train, size=(IMAGE_SIZE, IMAGE_SIZE))
test = tf.image.resize_images(test, size=(IMAGE_SIZE, IMAGE_SIZE))


def cross_entropy_loss(x, y):
    """
        x: real binarized data
        y: logits
    """
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=x, logits=y)


def mse_loss(x, y):
    return tf.losses.mean_squared_error(x, y)


inference_net = ResidualInferenceNetwork()
inference_net.compile(tf.train.AdamOptimizer(1e-3), loss=mse_loss)

inference_net.fit(x=train,
                  y=train,
                  batch_size=2,
                  epochs=5,
                  verbose=1,
                  callbacks=[tensorboard_callback])

test_predictions = inference_net.predict(test, batch_size=2, verbose=1)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(9, 9))
for ind, ax in enumerate(axes.flatten()):
    ax.imshow(test_predictions[ind].reshape(IMAGE_SIZE, IMAGE_SIZE, 3),
              vmin=0,
              vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.savefig('cifar10_test_image.png', dpi=50)