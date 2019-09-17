import tensorflow as tf


class InferenceNetwork(tf.keras.Model):
    def __init__(self, n_classes=3):
        super(InferenceNetwork, self).__init__()
        self.n_classes = n_classes
        # eg.: 128 x 128 x 3
        self.down_conv_1 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(3, 3),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 126 x 126 x 32
        self.downsample_1 = tf.keras.layers.Conv2D(filters=128,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 63 x 63 x 64
        self.down_conv_2 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 62 x 62 x 64
        self.downsample_2 = tf.keras.layers.Conv2D(filters=256,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 31 x 31 x 128
        self.down_conv_3 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(3, 3),
                                                  padding='valid',
                                                  activation='relu')
        # eg.: 29 x 29 x 128
        self.downsample_3 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 14 x 14 x 256
        self.bottom_conv1 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 13 x 13 x 256
        self.bottom_conv2 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   padding='valid',
                                                   activation='relu')
        # eg.: 12 x 12 x 256
        self.upsample_3 = tf.keras.layers.Conv2DTranspose(filters=512,
                                                          kernel_size=(7, 7),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 29 x 29 x 128
        self.up_conv_3_1 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg. : 28 x 28 x 128
        self.upsample_2 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                          kernel_size=(8, 8),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 62 x 62 x 64
        self.up_conv_2_1 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(2, 2),
                                                  padding='valid',
                                                  activation='relu')
        # eg. : 61 x 61 x 64
        self.upsample_1 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                          kernel_size=(6, 6),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg.:  126 x 126 x 32
        self.up_conv_1_1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                           kernel_size=(4, 4),
                                                           padding='valid',
                                                           activation='relu')
        # eg.: 129 x 129 x 32
        self.reco = tf.keras.layers.Conv2D(
            filters=self.n_classes,
            kernel_size=(2, 2),
            padding='valid',
            activation='sigmoid')  # scale to [0-1]

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
        print('After bottom conv 2 : ', x.get_shape().as_list())
        x = tf.contrib.layers.instance_norm(x)
        x = self.upsample_3(x)
        print('After upsampl 3 : ', x.get_shape().as_list())
        x = tf.keras.layers.Concatenate()([x, x3])
        x = self.up_conv_3_1(x)
        x = tf.contrib.layers.instance_norm(x)
        x = self.upsample_2(x)
        x = tf.keras.layers.Concatenate()([x, x2])
        x = self.up_conv_2_1(x)
        x = tf.contrib.layers.instance_norm(x)
        x = self.upsample_1(x)
        x = tf.keras.layers.Concatenate()([x, x1])
        x = self.up_conv_1_1(x)
        print('After up conv 1_1 : ', x.get_shape().as_list())
        x = tf.contrib.layers.instance_norm(x)
        x = self.reco(x)
        print('After reco : ', x.get_shape().as_list())
        return x