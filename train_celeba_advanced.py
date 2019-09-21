# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import pathlib
import glob
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from src import NetworkTrainer

BUFFER_SIZE = 50000
BATCH_SIZE = 4
WIDTH = 128
HEIGHT = 128

image = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3])
content_image = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3])
style = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 3])
model = NetworkTrainer(image, content_image, style)
sess = tf.Session()
sess.run(tf.initialize_all_variables())


def resize_image(img_path):
    img = plt.imread(img_path)
    img = resize(img, (WIDTH, HEIGHT))
    return img


all_image_paths = list(glob.glob("/home/qbeer666/celeba_small/*"))
images = np.array([resize_image(str(path)) for path in all_image_paths])
style_image = plt.imread('colorful_portrait.jpg')
style_image = np.expand_dims(resize(style_image, (WIDTH, HEIGHT)), axis=0)


def deprocess_input(img):
    """
        From keras_applications `caffee` mode
        Channels last format
    """
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255) / 255.
    return img


def save_fig(recos):
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(7, 4))
    for ind, ax in enumerate(axes.flatten()):
        ax.imshow(deprocess_input(recos[ind].reshape(WIDTH, HEIGHT, 3)),
                  vmin=0,
                  vmax=1)
    plt.tight_layout()
    plt.savefig('reco.png')
    plt.close(fig)


for _ in range(10):
    for ind in range(0, images.shape[0] - 4, 4):
        _, loss = sess.run(
            model.optimize, {
                image: images[ind:ind + 4],
                content_image: images[ind:ind + 4],
                style: style_image
            })
        print('LOSS : ', loss)
        if ind % 20 and ind > 0 == 0:
            recos, _, _, _ = sess.run(model.prediction, {
                image: images[:16],
                content_image: images[:16],
                style: style_image
            })
            save_fig(recos)