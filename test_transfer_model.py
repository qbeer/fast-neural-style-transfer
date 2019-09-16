# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()

from src import StyleTransferModel

transfer_model = StyleTransferModel()

x = tf.random.normal(shape=(1, 256, 256, 3))

logits, loss_out = transfer_model(x)

print("Logits shape : ", logits.numpy().shape)
print("Loss network outputs [VGG16 layers]: ", list(loss_out.keys()))