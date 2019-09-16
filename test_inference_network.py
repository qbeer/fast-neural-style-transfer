# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.datasets.mnist import load_data

from src import InferenceNetwork

(train, train_labels), (test, test_labels) = load_data()

# Binarize
train[train < .5] = 0
train[train > .5] = 1

print(train.shape)
train = tf.reshape(train, shape=(60_000, 28, 28, 1))

# Resize
train = tf.image.resize_images(train, size=(32, 32))
print(train.shape)


def cross_entropy_loss(x, y):
    """
        x: real binarized data
        y: logits
    """
    return -tf.losses.sigmoid_cross_entropy(multi_class_labels=x, logits=y)


inference_net = InferenceNetwork(n_classes=1)
inference_net.compile(tf.train.AdamOptimizer(1e-3), loss=cross_entropy_loss)

inference_net.fit(x=train, y=train, batch_size=256)
