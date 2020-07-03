import numpy as np
import matplotlib.pyplot as plt
import io
import tensorflow as tf


def deprocess_input(img):
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


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this __call__."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(reco):
    """Return a 2x2 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(8, 8))
    for i in range(4):
        # Start next subplot.
        plt.subplot(2, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(deprocess_input(reco[i]), cmap=plt.cm.binary)

    return figure