import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from src.networks import TransferModel
import matplotlib.pyplot as plt
from src.train_utils.imaging_utils import deprocess_input
import argparse
import glob


def sample(args):
    model = TransferModel((args.height, args.width, 3))
    model.build(input_shape=(None, args.height, args.width, 3))
    model.load_weights(args.model)

    img_paths = glob.glob(f'{args.img_path}/*.{args.img_ext}')

    for img_path in img_paths:
        img = plt.imread(img_path)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        img = tf.image.resize(img, size=(args.height, args.width))
        img = tf.expand_dims(img, axis=0)
        styled_img, _ = model(img)
        plt.imshow(deprocess_input(styled_img[0]))
        plt.axis('off')
        plt.savefig(img_path.replace(f'.{args.img_ext}', '_styled.png'),
                    dpi=80,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()


parser = argparse.ArgumentParser()

parser.add_argument('--img_ext', type=str, default='png', required=False)
parser.add_argument('--width', type=int, default=640, required=False)
parser.add_argument('--height', type=int, default=320, required=False)
parser.add_argument(
    '--model',
    default=False,
    type=str,
    required=True,
)
parser.add_argument(
    '--img_path',
    default='/home/qbeer/pics_vesuvio',
    type=str,
    required=False,
)

args = parser.parse_args()

sample(args)
