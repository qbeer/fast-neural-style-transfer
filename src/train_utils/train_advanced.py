import functools
import tensorflow as tf
from ..model import ResidualStyleTransferModel


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__

    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):

                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class NetworkTrainer:
    def __init__(self, image, content_image, style_image):
        self.image = image
        self.content_image = content_image
        self.style_image = style_image
        self.prediction
        self.optimize

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        x = self.image
        model = ResidualStyleTransferModel()
        model.loss_net.trainable = False
        reco, loss = model(x)
        prep_content_image = tf.keras.applications.vgg16.preprocess_input(
            self.content_image)
        content_loss = model.loss_net(prep_content_image)
        prep_style_image = tf.keras.applications.vgg16.preprocess_input(
            self.style_image)
        style_loss = model.loss_net(prep_style_image)
        return reco, loss, content_loss, style_loss

    @define_scope
    def optimize(self):
        reco, loss, content_loss, style_loss = self.prediction
        _, height, width, channels = content_loss['block5_conv2'].get_shape(
        ).as_list()
        feature_final_loss = tf.cast(
            1e3 / 2 * height * width * channels,
            dtype=tf.float32) * tf.reduce_sum(
                tf.square(content_loss['block5_conv2'] - loss['block5_conv2']))

        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in list(loss.keys()):
            gram_style = self._gram_matrix(style_loss[loss_layer])
            gram_reco = self._gram_matrix(loss[loss_layer])
            gram_diff = gram_reco - gram_style
            bs, height, width, channels = loss[loss_layer].get_shape().as_list(
            )
            style_final_loss += tf.cast(1e-3 * 0.166,
                                        dtype=tf.float32) * tf.reduce_sum(
                                            gram_diff**2)  # frob-norm
        total_var_loss = 1e-5 * tf.reduce_sum(tf.image.total_variation(reco))
        optimizer = tf.train.AdamOptimizer(1e-3)
        total_loss = style_final_loss + feature_final_loss + total_var_loss
        return optimizer.minimize(total_loss), total_loss

    def _gram_matrix(self, tensor):
        _, height, width, channels = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1, channels, height * width])
        tensor_T = tf.reshape(tensor, [-1, height * width, channels])
        return tf.matmul(tensor, tensor_T) / (2 * channels * width * height)
