import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import preprocessing
from config.constants import IMAGE_SIZE, CHANNELS

# Define the path to the model and the directory with test images
MODEL_DIR = 'models'
MODEL_VERSION = '1'  # Adjust based on the version you want to test
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_VERSION}.keras")
TEST_IMAGE_DIR = 'test'  # Directory with images for testing
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def predict (img):
    """Predict the class of an image"""
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    img_batch = np.expand_dims(img, 0)
    predictions = model.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

if __name__ == "__main__":
    print("Predicting the class of test images...")
    for img_name in os.listdir(TEST_IMAGE_DIR):
        img_path = os.path.join(TEST_IMAGE_DIR, img_name)
        img = preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = preprocessing.image.img_to_array(img)
        
        predicted_class, confidence = predict(img_array)
        
        plt.figure()
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}, Confidence: {confidence:.2%}")
        plt.axis("off")
        #SAVE THE FIGURE AS A PNG FILE
        plt.savefig(f"predictions/{img_name}")
