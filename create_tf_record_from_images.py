# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import glob

BUFFER_SIZE = 50000
BATCH_SIZE = 32
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 32


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_tf_example_from_image(image_string):
    feature = {
        'image': _bytes_feature(image_string),
        'height': _int64_feature(218),
        'width': _int64_feature(178),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


image_paths = glob.glob("/home/qbeer666/celeba/*")

with tf.io.TFRecordWriter('celeba.tfrecords') as writer:
    for image_path in image_paths:
        image_string = open(image_path, 'rb').read()
        tf_example = create_tf_example_from_image(image_string)
        writer.write(tf_example.SerializeToString())
