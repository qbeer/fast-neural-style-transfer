import tensorflow as tf


class OutputScale(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OutputScale, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x):
        return x * 255. / 2.

    def compute_output_shape(self, input_shape):
        return input_shape