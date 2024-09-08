from keras import layers, models
import tensorflow as tf
from model.augmentation import create_data_augmentation_pipeline

def build_model(input_shape, num_classes):
    """Builds the CNN model using an explicit Input layer."""
    resize_and_rescale, data_augmentation = create_data_augmentation_pipeline()

    inputs = tf.keras.Input(shape=input_shape)
    x = resize_and_rescale(inputs)
    x = data_augmentation(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    return model