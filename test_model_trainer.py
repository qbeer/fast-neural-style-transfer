# Tensorflow throws a bunch of `FutureWarning`s
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.enable_eager_execution()

from src import ModelTrainer

style_image = tf.random.normal(shape=(1, 256, 256, 3))

trainer = ModelTrainer(style_image=style_image, n_classes=3)

content_image = tf.random.poisson(lam=0.66, shape=(1, 256, 256, 3))

reco, loss = trainer.transfer_model(content_image)

style_loss, feature_loss = trainer._loss_fn(content_image=content_image,
                                            loss=loss)

print('Feature loss : ', feature_loss)
print('Style loss : ', style_loss)
