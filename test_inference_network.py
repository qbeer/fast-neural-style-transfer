# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.datasets.mnist import load_data
from tensorflow import keras

from src import InferenceNetwork

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

(train, train_labels), (test, test_labels) = load_data()

# Binarize
train[train < .5] = 0
train[train > .5] = 1
test[test < .5] = 0
test[test > .5] = 1

# Resize
train = tf.reshape(train, shape=(60_000, 28, 28, 1))
test = tf.reshape(test, shape=(10_000, 28, 28, 1))

train = tf.image.resize_images(train, size=(32, 32))
test = tf.image.resize_images(test, size=(32, 32))


def cross_entropy_loss(x, y):
    """
        x: real binarized data
        y: logits
    """
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=x, logits=y)


inference_net = InferenceNetwork(n_classes=1)
inference_net.compile(tf.train.AdamOptimizer(1e-3), loss=cross_entropy_loss)

inference_net.fit(x=train,
                  y=train,
                  batch_size=256,
                  epochs=10,
                  verbose=1,
                  callbacks=[tensorboard_callback])

test_predictions = inference_net.predict(test, batch_size=256, verbose=1)

import matplotlib.pyplot as plt

plt.imshow(test_predictions[0].reshape(32, 32), vmin=0, vmax=1, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.savefig('mnist_test_image.png', dpi=50)