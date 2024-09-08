import tensorflow as tf
from model import data_loader, tuner as tuner_module
from config.constants import IMAGE_SIZE, EPOCHS

def main():
    """Main function to run the training and hyperparameter tuning."""
    print("TensorFlow version:", tf.__version__)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Load and prepare the data
    dataset = data_loader.load_and_prepare_data()
    train_ds, val_ds, test_ds = data_loader.get_dataset_partitions_tf(dataset)
    train_ds, val_ds, test_ds = data_loader.prepare_datasets_for_training(train_ds, val_ds, test_ds)

    num_classes = len(dataset.class_names)
    input_shape = IMAGE_SIZE + (3,)

    # Configure the model and tuner
    hypermodel = tuner_module.get_hypermodel(input_shape, num_classes)
    tuner = tuner_module.get_tuner(hypermodel)

    # Perform hyperparameter search
    tuner.search(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Build and train the best model
    best_model = tuner.hypermodel.build(best_hps)
    best_model.summary()

    with tf.device('/GPU:0'):  # Ensure training runs on GPU
        best_model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds
        )

    # Save the best model
    best_model.save('best_model.h5')
    print("Model saved as 'best_model.h5'.")

if __name__ == "__main__":
    main()