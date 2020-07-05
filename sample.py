import tensorflow as tf
from src.networks import TransferModel
import matplotlib.pyplot as plt
from src.train_utils.imaging_utils import deprocess_input

model = TransferModel((160, 320, 3))
model.build(input_shape=(None, 160, 320, 3))
model.load_weights('model_2000.h5')

# Convert Van Gogh to other style
van_gogh = plt.imread('style_images/van_gogh.jpg')
van_gogh = tf.keras.applications.vgg16.preprocess_input(van_gogh)
van_gogh = tf.image.resize(van_gogh, size=(160, 320))
van_gogh = tf.expand_dims(van_gogh, axis=0)

styled_van_gogh, _ = model(van_gogh)

plt.imshow(deprocess_input(styled_van_gogh[0]))
plt.axis('off')
plt.savefig('data/styled_van_gogh.png',
            dpi=80,
            bbox_inches='tight',
            pad_inches=0)
plt.close()

# To make movie
import glob

img_paths = glob.glob('/home/qbeer/pics_vesuvio/*.png')

for img_path in img_paths:
    img = plt.imread(img_path)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = tf.image.resize(img, size=(160, 320))
    img = tf.expand_dims(img, axis=0)
    styled_img, _ = model(img)
    plt.imshow(deprocess_input(styled_img[0]))
    plt.axis('off')
    plt.savefig(img_path.replace('.png', '_styled.png'),
                dpi=80,
                bbox_inches='tight',
                pad_inches=0)
    plt.close()