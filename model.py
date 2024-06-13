import tensorflow as tf
from keras import models, layers
import matplotlib.pyplot as plt
import GPUtil
import threading

# Function to print GPU usage
def print_gpu_usage():
    GPUs = GPUtil.getGPUs()
    for gpu in GPUs:
        print(f"GPU ID: {gpu.id}, Name: {gpu.name}, Load: {gpu.load*100:.2f}%, Free Memory: {gpu.memoryFree}MB, Used Memory: {gpu.memoryUsed}MB, Total Memory: {gpu.memoryTotal}MB, Temperature: {gpu.temperature}°C")


# Start the thread to print GPU usage
monitor_thread = threading.Thread(target=print_gpu_usage, daemon=True)
monitor_thread.start()

# Verify
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Listar GPUs disponibles
print("GPUs: ", tf.config.experimental.list_physical_devices('GPU'))

# Constants
DIRECTORY = 'training'
SHUFFLE = True
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32

# Load the dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DIRECTORY,
    shuffle=SHUFFLE,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(class_names)

lenght = len(dataset)

# Divide the dataset into training, validation and test
train_size = int(0.8 * lenght)
val_size = int(0.1 * lenght)
test_size = int(0.1 * lenght)


