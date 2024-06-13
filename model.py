import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import GPUtil
import threading

# Constants
DIRECTORY = 'training'
SHUFFLE = True
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SHUFFLE_SIZE = 10000
SEED = 12

def print_gpu_usage():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load*100:.2f}%, Free Memory: {gpu.memoryFree}MB, Used Memory: {gpu.memoryUsed}MB, Total Memory: {gpu.memoryTotal}MB, Temperature: {gpu.temperature}°C")

def start_gpu_monitoring():
    monitor_thread = threading.Thread(target=print_gpu_usage, daemon=True)
    monitor_thread.start()

def get_dataset_partitions_tf(ds, train_split=TRAIN_SPLIT, val_split=VAL_SPLIT, test_split=TEST_SPLIT, shuffle=SHUFFLE, shuffle_size=SHUFFLE_SIZE, seed=SEED):
    assert (train_split + val_split + test_split) == 1, "Splits must sum to 1"

    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=seed)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)

    return train_ds, val_ds, test_ds

def load_and_prepare_data(directory, image_size, batch_size, shuffle):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        shuffle=shuffle,
        image_size=image_size,
        batch_size=batch_size
    )
    class_names = dataset.class_names
    print("Class names:", class_names)
    return dataset

def main():
    start_gpu_monitoring()

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

    dataset = load_and_prepare_data(DIRECTORY, IMAGE_SIZE, BATCH_SIZE, SHUFFLE)

    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")

if __name__ == "__main__":
    main()