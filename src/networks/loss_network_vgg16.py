import tensorflow as tf


class LossNetwork(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3)):
        super(LossNetwork, self).__init__()
        self.base_model = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_shape=input_shape)
        for layer in self.base_model.layers:
            layer.trainable = False

    def call(self, x):
        outputs = {
            "block2_conv2": None,  # feature
            "block1_conv2": None,  # style
            "block2_conv2": None,  # style
            "block3_conv3": None,  # style
            "block4_conv3": None,  # style
        }
        for layer in self.base_model.layers:
            x = layer(x)
            if layer.name in list(outputs.keys()):
                outputs[layer.name] = x
        return outputs
