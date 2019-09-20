import tensorflow as tf
from ..networks import LossNetwork, ResidualInferenceNetwork


class ResidualStyleTransferModel(tf.keras.Model):
    def __init__(self, input_shape=(256, 256, 3), n_classes=3):
        super(ResidualStyleTransferModel, self).__init__()
        self.inference_net = ResidualInferenceNetwork()
        self.loss_net = LossNetwork(input_shape=input_shape)
        self.loss_net.trainable = False

    def call(self, x):
        x = tf.keras.applications.vgg16.preprocess_input(x)
        reco = self.inference_net(x)
        loss = self.loss_net(reco)
        return reco, loss
