import tensorflow as tf
from .loss_network_vgg16 import LossNetwork
from .original_inference_network import ResidualInferenceNetwork


class TransferModel(tf.keras.Model):
    def __init__(self, input_sape):
        super(TransferModel, self).__init__()
        self.inference_net = ResidualInferenceNetwork()
        self.loss_net = LossNetwork(input_shape=input_sape)

    def __call__(self, x):
        reco = self.inference_net(x)
        loss = self.loss_net(reco)
        return reco, loss
