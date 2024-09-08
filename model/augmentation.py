from keras import layers
import tensorflow as tf
from config.constants import IMAGE_SIZE

def create_data_augmentation_pipeline():
    """Creates a data augmentation pipeline."""
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        layers.Rescaling(1./255)
    ])

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2)
    ])

    return resize_and_rescale, data_augmentation