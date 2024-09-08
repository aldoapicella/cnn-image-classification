from kerastuner import HyperModel, RandomSearch
import tensorflow as tf
from keras import layers
from model.augmentation import create_data_augmentation_pipeline
from config.constants import IMAGE_SIZE, MAX_TRIALS, EXECUTION_PER_TRIAL

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        """Builds the hypermodel with hyperparameter tuning."""
        resize_and_rescale, data_augmentation = create_data_augmentation_pipeline()

        model = tf.keras.Sequential([
            resize_and_rescale,
            data_augmentation,
            layers.Conv2D(
                filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
                activation='relu',
                input_shape=self.input_shape
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(
                filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
                activation='relu'
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(
                units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
                activation='relu'
            ),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        return model

def get_hypermodel(input_shape, num_classes):
    """Returns the hypermodel."""
    return CNNHyperModel(input_shape, num_classes)

def get_tuner(hypermodel, objective='val_accuracy'):
    """Returns the tuner for hyperparameter optimization."""
    return RandomSearch(
        hypermodel,
        objective=objective,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='hyperband',
        project_name='image_classification'
    )