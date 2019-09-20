from ..model import StyleTransferModel, ResidualStyleTransferModel
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class ModelTrainer:
    def __init__(self,
                 style_image,
                 residual=False,
                 batch_size=32,
                 input_shape=(256, 256, 3),
                 n_classes=3):
        if residual:
            self.transfer_model = ResidualStyleTransferModel()
        else:
            self.transfer_model = StyleTransferModel(input_shape=input_shape,
                                                     n_classes=n_classes)
        self.transfer_model.loss_net.trainable = False
        prep_style_image = tf.keras.applications.vgg16.preprocess_input(
            style_image)
        self.style_loss = self.transfer_model.loss_net(prep_style_image)
        self.batch_size = batch_size

    def _loss_fn(self, content_image, reco, loss):
        prep_content_image = tf.keras.applications.vgg16.preprocess_input(
            content_image)
        content_loss = self.transfer_model.loss_net(prep_content_image)
        """
        block5_conv2   # for style
        block2_conv1   # for feature & style
        block1_conv1   # for style
        block3_conv1   # for style
        block4_conv1   # for style
        block5_conv1   # for style
        """
        bs, height, width, channels = content_loss['block2_conv1'].get_shape(
        ).as_list()
        feature_final_loss = 1e2 / tf.cast(
            height * width * channels * bs, dtype=tf.float32) * tf.reduce_sum(
                tf.square(content_loss['block2_conv1'] - loss['block2_conv1']))

        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in list(loss.keys()):
            gram_style = self._gram_matrix(self.style_loss[loss_layer])
            gram_reco = self._gram_matrix(loss[loss_layer])
            gram_diff = gram_reco - gram_style
            bs, height, width, channels = loss[loss_layer].get_shape().as_list(
            )
            style_final_loss += 1e-3 * tf.reduce_sum(gram_diff**2)  # frob-norm
        total_var_loss = 1e-1 * tf.reduce_sum(tf.image.total_variation(reco))
        self.style_loss_f.write("%.5f\n" % style_final_loss.numpy())
        self.feature_loss_f.write("%.5f\n" % feature_final_loss.numpy())
        self.total_variation_loss_f.write("%.5f\n" % total_var_loss.numpy())
        total_loss = style_final_loss + feature_final_loss + total_var_loss
        self.total_loss_f.write("%.5f\n" % total_loss.numpy())
        return total_loss

    def _gram_matrix(self, tensor):
        bs, height, width, channels = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1, channels, height * width])
        tensor_T = tf.reshape(tensor, [-1, height * width, channels])
        return tf.matmul(tensor,
                         tensor_T) / (2 * channels * width * height * bs)

    def _train_step(self, image_batch):
        with tf.GradientTape() as inference_net_tape:
            reco, loss = self.transfer_model(image_batch)
            full_loss = self._loss_fn(image_batch, reco, loss)

        gradients = inference_net_tape.gradient(
            full_loss, self.transfer_model.inference_net.trainable_variables)

        self.opt.apply_gradients(
            zip(gradients,
                self.transfer_model.inference_net.trainable_variables))

    def train(self, images, lr=1e-2, epochs=1):
        self.opt = tf.train.RMSPropOptimizer(lr)

        self.total_loss_f = open('total_loss.txt', "a+")
        self.feature_loss_f = open('feature_loss.txt', "a+")
        self.style_loss_f = open('style_loss.txt', "a+")
        self.total_variation_loss_f = open('total_variation_loss.txt', "a+")

        for epoch in range(epochs):
            for ind, image_batch in enumerate(images):
                self._train_step(image_batch)
                if ind % 20 == 0:
                    self._save_fig(image_batch)
                    self._save_stats()
                    # Re-open
                    self.total_loss_f = open('total_loss.txt', "a+")
                    self.feature_loss_f = open('feature_loss.txt', "a+")
                    self.style_loss_f = open('style_loss.txt', "a+")
                    self.total_variation_loss_f = open(
                        'total_variation_loss.txt ', "a+")

        self.transfer_model.save_weights("model.h5")

    def _save_fig(self, image_batch):
        reco, _ = self.transfer_model(image_batch)
        fig, axes = plt.subplots(2,
                                 2,
                                 sharex=True,
                                 sharey=True,
                                 figsize=(7, 7))

        for ind, ax in enumerate(axes.flatten()):
            ax.imshow(self._deprocess_input(reco[ind]), vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.savefig('reco.png')
        plt.close(fig)

    def _save_stats(self):
        self.total_loss_f.close()
        self.feature_loss_f.close()
        self.style_loss_f.close()
        self.total_variation_loss_f.close()
        total_loss = np.loadtxt('total_loss.txt', delimiter='\n')
        feature_loss = np.loadtxt('feature_loss.txt', delimiter='\n')
        style_loss = np.loadtxt('style_loss.txt', delimiter='\n')
        TV_loss = np.loadtxt('total_variation_loss.txt', delimiter='\n')
        fig = plt.figure(figsize=(12, 10))
        plt.subplot('221')
        plt.plot(total_loss, 'r--')
        plt.title('Total loss')
        plt.subplot('222')
        plt.title('Feature loss')
        plt.plot(feature_loss, 'b--')
        plt.subplot(223)
        plt.title('Style loss')
        plt.plot(style_loss, 'g--')
        plt.subplot(224)
        plt.title('Style and feature losses')
        plt.plot(style_loss, 'g--')
        plt.plot(feature_loss, 'b--')
        plt.savefig('loss.png')
        plt.close(fig)

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
        img = np.clip(img, 0, 255) / 255.
        return img