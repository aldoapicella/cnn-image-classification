# model/model.py

from keras import layers, models
import tensorflow as tf
from model.augmentation import create_data_augmentation_pipeline

def build_model(input_shape, num_classes):
    """Builds the CNN model."""
    resize_and_rescale, data_augmentation = create_data_augmentation_pipeline()
    
    model = models.Sequential([
        resize_and_rescale,
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model