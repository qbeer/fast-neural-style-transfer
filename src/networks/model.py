import tensorflow as tf
from .loss_network_vgg16 import LossNetwork
from .original_inference_network import ResidualInferenceNetwork


class TransferModel(tf.keras.Model):
    def __init__(self, input_sape):
        super(TransferModel, self).__init__()
        self.inference_net = ResidualInferenceNetwork()
        self.loss_net = LossNetwork(input_shape=input_sape)

    def call(self, x, training=True):
        reco = self.inference_net(x, training)
        loss = self.loss_net(reco, training)
        return reco, loss
