from ..model import StyleTransferModel
import tensorflow as tf


class ModelTrainer:
    def __init__(self, style_image, input_shape=(256, 256, 3), n_classes=3):
        self.transfer_model = StyleTransferModel(input_shape=input_shape,
                                                 n_classes=n_classes)
        self.style_loss = self.transfer_model.loss_net(style_image)

    def _loss_fn(self, content_image, loss):
        content_loss = self.transfer_model.loss_net(content_image)
        """
        block2_conv2   (1, 128, 128, 128) # for style and feature
        block1_conv2   (1, 256, 256, 64)
        block3_conv3   (1, 64, 64, 256)
        block4_conv3   (1, 32, 32, 512)
        """
        feature_final_loss = tf.reduce_mean(
            tf.square(content_loss['block2_conv2'] - loss['block2_conv2']))

        print('Feature loss : ', feature_final_loss)

        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in list(loss.keys()):
            print('loss layer : ', loss_layer)
            channels = tf.cast(tf.shape(
                self.style_loss[loss_layer]).numpy()[-1],
                               dtype=tf.int32)
            area = tf.cast(
                tf.reduce_prod(tf.shape(self.style_loss[loss_layer])).numpy() /
                channels,
                dtype=tf.int32)
            phi_style = tf.reshape(self.style_loss[loss_layer],
                                   shape=(channels, area))
            gram_style = tf.matmul(phi_style, phi_style, transpose_b=True)
            phi_reco = tf.reshape(loss[loss_layer], shape=(channels, area))
            gram_reco = tf.matmul(phi_reco, phi_reco, transpose_b=True)
            style_final_loss += tf.square(
                tf.cast(1. / tf.cast(area * channels, dtype=tf.float32),
                        dtype=tf.float32)) * tf.reduce_sum(
                            tf.square(gram_reco - gram_style))
            print('style loss : ', style_final_loss)
        return style_final_loss, feature_final_loss

    def _train_step(self, image, opt):
        with tf.GradientTape() as inference_net_tape:
            _, loss = self.transfer_model(image)
            style_loss, feature_loss = self._loss_fn(image, loss)

        gradients = inference_net_tape.gradient(
            style_loss + feature_loss,
            self.transfer_model.inference_net.variables)

        opt.apply_gradients(
            zip(gradients, self.transfer_model.inference_net.variables))

    def train(self, images, lr=1e-3, epochs=10):
        opt = tf.train.AdamOptimizer(lr)
        for epoch in range(epochs):
            for image in images:
                self._train_step(image, opt)
