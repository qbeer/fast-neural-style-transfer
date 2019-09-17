from ..model import StyleTransferModel
import tensorflow as tf
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self,
                 style_image,
                 batch_size=32,
                 input_shape=(256, 256, 3),
                 n_classes=3):
        self.transfer_model = StyleTransferModel(input_shape=input_shape,
                                                 n_classes=n_classes)
        self.style_loss = self.transfer_model.loss_net(style_image)
        self.batch_size = batch_size

    def _loss_fn(self, content_image, reco, loss):
        content_loss = self.transfer_model.loss_net(content_image)
        """
        block2_conv2   (1, 128, 128, 128) # for style and feature
        block1_conv2   (1, 256, 256, 64)
        block3_conv3   (1, 64, 64, 256)
        block4_conv3   (1, 32, 32, 512)
        """
        channels = tf.cast(tf.shape(content_loss['block2_conv2'])[-1],
                           dtype=tf.int32)
        area = tf.cast(tf.reduce_prod(tf.shape(content_loss['block2_conv2'])) /
                       (channels * self.batch_size),
                       dtype=tf.int32)
        feature_final_loss = 1. / tf.cast(
            area * channels * self.batch_size,
            dtype=tf.float32) * tf.reduce_sum(
                tf.square(content_loss['block2_conv2'] - loss['block2_conv2']))

        style_final_loss = tf.cast(0., dtype=tf.float32)
        for loss_layer in list(loss.keys()):
            gram_style = self._gram_matrix(self.style_loss[loss_layer])
            gram_reco = self._gram_matrix(loss[loss_layer])
            gram_diff = gram_reco - gram_style
            style_final_loss += 10. * tf.reduce_sum(gram_diff**2)  # frob-norm
        print('Style : ', style_final_loss)
        print('Feature : ', feature_final_loss)
        #return feature_final_loss + tf.reduce_sum(
        #    tf.square(reco - content_image))
        return style_final_loss + feature_final_loss

    def _gram_matrix(self, tensor):
        bs, height, width, channels = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [bs, channels, height * width])
        tensor_T = tf.reshape(tensor, [bs, height * width, channels])
        return tf.matmul(tensor, tensor_T) / (channels * width * height * bs)

    def _train_step(self, image_batch, opt):
        with tf.GradientTape() as inference_net_tape:
            reco, loss = self.transfer_model(image_batch)
            full_loss = self._loss_fn(image_batch, reco, loss)

        print('Loss : ', full_loss)
        print('\n')

        gradients = inference_net_tape.gradient(
            full_loss, self.transfer_model.inference_net.trainable_variables)

        opt.apply_gradients(
            zip(gradients,
                self.transfer_model.inference_net.trainable_variables))

    def train(self, images, lr=1e-2, epochs=10):
        opt = tf.train.AdamOptimizer(lr)
        for epoch in range(epochs):
            for ind, image_batch in enumerate(images):
                self._train_step(image_batch, opt)
                if ind % 10 == 0:
                    reco, loss = self.transfer_model(image_batch)
                    plt.subplot('121')
                    plt.imshow(reco[0])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.subplot('122')
                    plt.imshow(image_batch[0])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig("reco.png")
                    plt.close()
