import tensorflow as tf


class InferenceNetwork(tf.keras.Model):
    def __init__(self, n_classes=3):
        super(InferenceNetwork, self).__init__()
        self.n_classes = n_classes
        # eg.: 256 x 256 x 3
        self.down_conv_1 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.downsample_1 = tf.keras.layers.Conv2D(filters=64,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   padding='same',
                                                   activation='relu')
        # eg.: 128 x 128 x 64
        self.down_conv_2 = tf.keras.layers.Conv2D(filters=256,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.downsample_2 = tf.keras.layers.Conv2D(filters=256,
                                                   kernel_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='same',
                                                   activation='relu')
        # eg.: 64 x 64 x 256
        self.down_conv_3 = tf.keras.layers.Conv2D(filters=512,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.downsample_3 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='same',
                                                   activation='relu')
        # eg.: 32 x 32 x 512
        self.bottom_conv1 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(2, 2),
                                                   padding='same',
                                                   activation='relu')
        # eg.: 32 x 32 x 512
        self.bottom_conv2 = tf.keras.layers.Conv2D(filters=512,
                                                   kernel_size=(1, 1),
                                                   padding='same',
                                                   activation='relu')
        # eg.: 32 x 32 x 512
        self.upsample_3 = tf.keras.layers.Conv2DTranspose(filters=256,
                                                          kernel_size=(2, 2),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 64 x 64 x 512
        self.up_conv_3_1 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.upsample_2 = tf.keras.layers.Conv2DTranspose(filters=256,
                                                          kernel_size=(2, 2),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 128 x 128 x 256
        self.up_conv_2_1 = tf.keras.layers.Conv2D(filters=128,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.upsample_1 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                          kernel_size=(2, 2),
                                                          strides=(2, 2),
                                                          padding='valid',
                                                          activation='relu')
        # eg. : 256 x 256 x 64
        self.up_conv_1_1 = tf.keras.layers.Conv2D(filters=64,
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='relu')
        self.logits = tf.keras.layers.Conv2D(filters=self.n_classes,
                                             kernel_size=(2, 2),
                                             padding='same')

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
        x = self.upsample_3(x)
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
        x = tf.contrib.layers.instance_norm(x)
        return self.logits(x)