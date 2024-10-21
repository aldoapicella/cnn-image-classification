import os
import tensorflow as tf
import matplotlib.pyplot as plt
from model import data_loader, model as model_module
from config.constants import IMAGE_SIZE, EPOCHS, BATCH_SIZE, CHANNELS
from utils.plotting import plot_training_history
from utils.inference import predict

def main():
    """Main function to run the training."""
    print("TensorFlow version:", tf.__version__)

    # Limitar el uso de memoria de GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Load and prepare the data
    dataset = data_loader.load_and_prepare_data()
    train_ds, val_ds, test_ds = data_loader.get_dataset_partitions_tf(dataset)
    train_ds, val_ds, test_ds = data_loader.prepare_datasets_for_training(train_ds, val_ds, test_ds)

    # Calculate number of classes from dataset
    class_names = dataset.class_names
    num_classes = len(class_names)
    input_shape = IMAGE_SIZE + (CHANNELS,)

    # Build and train the model
    cnn_model = model_module.build_model(input_shape, num_classes)
    cnn_model.summary()

    with tf.device('/GPU:0'):  # Ensure training runs on GPU
        history = cnn_model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            batch_size=BATCH_SIZE
        )

    # Plot training history
    plot_training_history(history, EPOCHS)

    # Run prediction on a sample image
    plt.figure(figsize=(15, 15))
    for images, labels in test_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))

            predicted_class, confidence = predict(cnn_model, images[i].numpy(), class_names)
            actual_class = class_names[labels[i]]

            plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
            plt.axis("off")
    
    # Save the figure as a PNG file
    plt.savefig("predictions.png")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_version = max([int(i) for i in os.listdir(model_dir) if i.isdigit()] + [0]) + 1
    model_save_path = os.path.join(model_dir, f"{model_version}.keras")
    cnn_model.save(model_save_path)
    print(f"Model saved as version {model_version} at '{model_save_path}'.")

if __name__ == "__main__":
    main()