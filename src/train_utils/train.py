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
        self.prep_style_image = tf.keras.applications.vgg16.preprocess_input(
            style_image)
        self.style_loss = self.transfer_model.loss_net(self.prep_style_image)
        self.batch_size = batch_size

    def _loss_fn(self, content_image, reco, loss):
        prep_content_image = tf.keras.applications.vgg16.preprocess_input(
            content_image)
        print(np.min(prep_content_image), np.max(prep_content_image))
        print(np.min(self.prep_style_image), np.max(self.prep_style_image))
        print(np.min(reco), np.max(reco), '\n')
        content_loss = self.transfer_model.loss_net(prep_content_image)
        """
        block5_conv2   # for style
        block2_conv1   # for feature & style
        block1_conv1   # for style
        block3_conv1   # for style
        block4_conv1   # for style
        block5_conv1   # for style
        """
        bs, height, width, channels = content_loss['block4_conv2'].get_shape(
        ).as_list()
        feature_final_loss = 2e4 / tf.cast(
            2 * height * width * channels * bs,
            dtype=tf.float32) * tf.reduce_sum(
                tf.square(
                    tf.norm(content_loss['block4_conv2'] -
                            loss['block4_conv2'])))

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
            style_final_loss += 5e-2 * tf.reduce_sum(
                tf.square(gram_diff))  # frob-norm
        total_var_loss = 0 * tf.reduce_mean(tf.image.total_variation(reco))
        self.style_loss_f.write("%.5f\n" % style_final_loss.numpy())
        self.feature_loss_f.write("%.5f\n" % feature_final_loss.numpy())
        self.total_variation_loss_f.write("%.5f\n" % total_var_loss.numpy())
        total_loss = style_final_loss + feature_final_loss + total_var_loss
        self.total_loss_f.write("%.5f\n" % total_loss.numpy())
        return total_loss

    def _gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(
            2 * input_shape[1] * input_shape[2] * input_shape[0] *
            input_shape[3], tf.float32)
        return result / (num_locations)

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
        self.opt = tf.train.AdamOptimizer(learning_rate=lr,
                                          beta1=0.99,
                                          epsilon=1e-1)

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
        #plt.plot(feature_loss + style_loss, 'b--', alpha=0.4)
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