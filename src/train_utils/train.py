from ..networks import ResidualInferenceNetwork, LossNetwork, InferenceNetwork, TransferModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io
from .imaging_utils import image_grid, plot_to_image


class ModelTrainer:
    def __init__(self,
                 style_image,
                 residual=False,
                 batch_size=32,
                 c_out=3,
                 input_shape=(256, 256, 3)):
        if residual:
            self.transfer_model = TransferModel(input_sape=input_shape)
        else:
            self.transfer_model = InferenceNetwork(c_out=c_out)

        self.batch_size = batch_size
        self.style_image = style_image

    def _style_loss(self, loss):
        style_loss = self.transfer_model.loss_net(self.style_image)

        STYLE_LAYERS = [
            'block3_conv3', 'block2_conv2', 'block3_conv3', 'block4_conv3'
        ]
        layer_weight = 1. / float(len(STYLE_LAYERS))
        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in STYLE_LAYERS:
            gram_style = self._gram_matrix(style_loss[loss_layer])
            gram_reco = self._gram_matrix(loss[loss_layer])
            gram_diff = gram_reco - gram_style
            bs, height, width, channels = loss[loss_layer].get_shape().as_list(
            )
            style_final_loss += layer_weight * tf.reduce_sum(
                tf.square(gram_diff))  # frob-norm
        return style_final_loss

    def _content_loss(self, content_image, loss):
        CONTENT_LAYER = 'block2_conv2'
        content_loss = self.transfer_model.loss_net(content_image)
        bs, height, width, channels = content_loss[CONTENT_LAYER].get_shape(
        ).as_list()
        content_final_loss = tf.reduce_sum(
            tf.square(content_loss[CONTENT_LAYER] -
                      loss[CONTENT_LAYER])) / (height * width * channels)
        return content_final_loss

    def _loss_fn(self, content_image, reco, loss):
        feature_final_loss = 400. * self._content_loss(content_image, loss)
        style_final_loss = 0.01 * self._style_loss(loss)
        total_var_loss = 1e-2 * tf.reduce_mean(tf.image.total_variation(reco))
        total_loss = style_final_loss + feature_final_loss + total_var_loss

        return total_loss, style_final_loss, feature_final_loss, total_var_loss

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(
            input_shape[1] * input_shape[2] * input_shape[3], tf.float32)
        return result / num_locations

    def _train_step(self, image_batch):
        with tf.GradientTape() as transfer_net_tape:
            reco, loss = self.transfer_model(image_batch)

            full_loss, style_loss, feature_loss, TV_loss = self._loss_fn(
                image_batch, reco, loss)

        gradients = transfer_net_tape.gradient(
            full_loss, self.transfer_model.inference_net.trainable_variables)

        self.opt.apply_gradients(
            zip(gradients,
                self.transfer_model.inference_net.trainable_variables))

        return reco, full_loss, style_loss, feature_loss, TV_loss

    def train(self, images, lr=1e-2, epochs=1):
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        _it = 0

        for epoch in range(epochs):
            for ind, image_batch in enumerate(images):
                reco, full_loss, style_loss, feature_loss, TV_loss = self._train_step(
                    image_batch)

                tf.summary.scalar('full_loss', full_loss, step=_it)
                tf.summary.scalar('style_loss', style_loss, step=_it)
                tf.summary.scalar('TV_loss', TV_loss, step=_it)
                tf.summary.scalar('feature_loss', feature_loss, step=_it)
                tf.summary.scalar('reco_min',
                                  tf.reduce_mean(
                                      tf.reduce_min(reco, axis=[1, 2, 3])),
                                  step=_it)
                tf.summary.scalar('reco_max',
                                  tf.reduce_mean(
                                      tf.reduce_max(reco, axis=[1, 2, 3])),
                                  step=_it)

                figure = image_grid(reco)

                tf.summary.image("test image", plot_to_image(figure), step=_it)

                _it += 1

                if _it % 1000 == 0:
       	            self.transfer_model.save_weights(f"model_{_it}.h5")
