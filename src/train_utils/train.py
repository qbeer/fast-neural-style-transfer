from ..model import StyleTransferModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainer:
    def __init__(self,
                 style_image,
                 batch_size=32,
                 input_shape=(256, 256, 3),
                 n_classes=3):
        self.transfer_model = StyleTransferModel(input_shape=input_shape,
                                                 n_classes=n_classes)
        prep_style_image = tf.keras.applications.vgg16.preprocess_input(
            style_image)
        self.style_loss = self.transfer_model.loss_net(prep_style_image)
        self.batch_size = batch_size

    def _loss_fn(self, content_image, reco, loss):
        prep_content_image = tf.keras.applications.vgg16.preprocess_input(
            content_image)
        content_loss = self.transfer_model.loss_net(prep_content_image)
        """
        block5_conv2   # for feature
        block2_conv1   # for style
        block1_conv1   # for style
        block3_conv1   # for style
        block4_conv1   # for style
        """
        bs, height, width, channels = content_loss['block5_conv2'].get_shape(
        ).as_list()
        feature_final_loss = 1e3 / tf.cast(
            2 * height * width * channels * self.batch_size,
            dtype=tf.float32) * tf.reduce_sum(
                tf.square(content_loss['block5_conv2'] - loss['block5_conv2']))

        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in [
                'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1',
                'block5_conv1'
        ]:
            gram_style = self._gram_matrix(self.style_loss[loss_layer])
            gram_reco = self._gram_matrix(loss[loss_layer])
            gram_diff = gram_reco - gram_style
            bs, height, width, channels = loss[loss_layer].get_shape().as_list(
            )
            style_final_loss += (1e-2 / 5) * tf.reduce_sum(gram_diff**
                                                           2)  # frob-norm
        total_var_loss = 1e-4 * tf.reduce_sum(tf.image.total_variation(reco))
        print('Style : ', style_final_loss)
        print('Feature : ', feature_final_loss)
        print('TV loss : ', total_var_loss)
        return style_final_loss + feature_final_loss + total_var_loss

    def _gram_matrix(self, tensor):
        bs, height, width, channels = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [bs, channels, height * width])
        tensor_T = tf.reshape(tensor, [bs, height * width, channels])
        return tf.matmul(tensor,
                         tensor_T) / (2 * channels * width * height * bs)

    def _train_step(self, image_batch):
        with tf.GradientTape() as inference_net_tape:
            reco, loss = self.transfer_model(image_batch)
            full_loss = self._loss_fn(image_batch, reco, loss)

        print('Loss : ', full_loss)
        print('\n')

        gradients = inference_net_tape.gradient(
            full_loss, self.transfer_model.inference_net.trainable_variables)

        self.opt.apply_gradients(
            zip(gradients,
                self.transfer_model.inference_net.trainable_variables))

    def train(self, images, lr=1e-2, epochs=1):
        self.opt = tf.train.AdamOptimizer(lr, beta1=0.99, epsilon=.1)
        for epoch in range(epochs):
            for ind, image_batch in enumerate(images):
                self._train_step(image_batch)
                self._save_fig(image_batch, ind)
        self.transfer_model.save_weights("model.h5")

    def _save_fig(self, image_batch, ind):
        if ind % 10 == 0:
            reco, loss = self.transfer_model(image_batch)
            plt.subplot('121')
            plt.imshow(self._deprocess_input(reco[0]))
            plt.xticks([])
            plt.yticks([])
            plt.subplot('122')
            plt.imshow(image_batch[0] / 255.)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig("reco.png")
            plt.close()

    def _deprocess_input(self, img):
        """
            From keras_applications `caffee` mode
        """
        img = img.numpy()
        """
            Channels last format
        """
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

        img = img[:, :, ::-1]
        img = np.clip(img, 0, 255).astype(int)
        return img