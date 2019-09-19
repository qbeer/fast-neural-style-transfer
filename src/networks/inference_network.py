import tensorflow as tf


class InferenceNetwork(tf.keras.Model):
    def __init__(self, n_classes=3):
        super(InferenceNetwork, self).__init__()
        self.n_classes = n_classes
        # eg.: 128 x 128 x 3
        self.down_conv_1 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 127 x 127 x 64
        self.downsample_1 = tf.keras.layers.Conv2D(filters=128,
                                                   kernel_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 63 x 63 x 128
        self.down_conv_2 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 62 x 62 x 128
        self.downsample_2 = tf.keras.layers.Conv2D(filters=256,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 31 x 31 x 256
        self.down_conv_3 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 30 x 30 x 256
        self.downsample_3 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 15 x 15 x 512
        self.bottom_conv1 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 13 x 13 x 256
        self.bottom_conv2 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   padding='valid',
                                                   activation='relu')

        self.bili3_conv = tf.keras.layers.Conv2D(256,
                                                 kernel_size=(4, 4),
                                                 activation='relu',
                                                 padding='valid')

        self.upsample_3_bili = tf.keras.layers.UpSampling2D(
            size=(3, 3), interpolation='bilinear')

        # eg.: 12 x 12 x 512
        self.upsample_3 = tf.keras.layers.Conv2DTranspose(filters=256,
                                                          kernel_size=(6, 6),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 30 x 30 x 256
        self.up_conv_3_1 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')

        # eg. : 29 x 29 x 256
        self.upsample_2 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                          kernel_size=(6, 6),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 64 x 64 x 128
        self.up_conv_2_1 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')

        # eg. : 63 x 63 x 128
        self.bili1_conv = tf.keras.layers.Conv2D(64,
                                                 kernel_size=(2, 2),
                                                 activation='relu',
                                                 padding='valid')

        # eg. : 63 x 63 x 128
        self.upsample_1_bili = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation='bilinear')

        # eg. : 63 x 63 x 128
        self.upsample_1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                          kernel_size=(7, 7),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')

        # eg. : 127 x 127 x 64
        self.up_conv_1_1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(3, 3),
                                                           padding='valid',
                                                           activation='relu')
        # eg.: 129 x 129 x 64
        self.reco = tf.keras.layers.Conv2D(filters=self.n_classes,
                                           kernel_size=(2, 2),
                                           padding='valid')

    def call(self, x):
        x = self.down_conv_1(x)
        x = tf.contrib.layers.instance_norm(x)
        x1 = x
        x = self.downsample_1(x)
        x = self.down_conv_2(x)
        x = tf.contrib.layers.instance_norm(x)
        x2 = x
        x = self.downsample_2(x)
        x = self.down_conv_3(x)
        x = tf.contrib.layers.instance_norm(x)
        x3 = x
        x = self.downsample_3(x)
        x = self.bottom_conv1(x)
        x = tf.contrib.layers.instance_norm(x)
        x = self.bottom_conv2(x)
        x = tf.contrib.layers.instance_norm(x)
        x_3 = self.bili3_conv(x)
        x_3 = self.upsample_3_bili(x_3)
        x = self.upsample_3(x)
        x = tf.keras.layers.Concatenate()([x, x3, x_3])
        x = self.up_conv_3_1(x)
        x = tf.contrib.layers.instance_norm(x)
        x = self.upsample_2(x)
        x = tf.keras.layers.Concatenate()([x, x2])
        x = self.up_conv_2_1(x)
        x = tf.contrib.layers.instance_norm(x)
        x_1 = self.upsample_1_bili(x)
        x_1 = self.bili1_conv(x_1)
        x = self.upsample_1(x)
        paddings = [[0, 0], [3, 3], [3, 3], [0, 0]]
        x = tf.keras.layers.Concatenate()(
            [x, x1, tf.pad(x_1, paddings, "CONSTANT")])
        x = self.up_conv_1_1(x)
        x = tf.contrib.layers.instance_norm(x)
        print(x.get_shape().as_list())
        x = self.reco(x)
        return x