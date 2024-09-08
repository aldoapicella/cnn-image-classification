# Image Classification Project with TensorFlow and Keras Tuner

This project demonstrates how to build and tune a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras Tuner. It features a modular structure for efficient management and scalability.

## Project Structure

```
project/
│
├── config/
│   └── constants.py
│
├── model/
│   ├── data_loader.py
│   ├── gpu_monitor.py
│   ├── augmentation.py
│   ├── model.py
│   └── tuner.py
│
├── main.py
└── README.md
```

- **config/constants.py**: Contains configurable constants used across the project.
- **model/data_loader.py**: Handles data loading and preprocessing.
- **model/gpu_monitor.py**: Monitors GPU usage during training.
- **model/augmentation.py**: Defines data augmentation pipelines.
- **model/model.py**: Constructs the CNN model.
- **model/tuner.py**: Implements hyperparameter tuning using Keras Tuner.
- **main.py**: The main script to run the model training and hyperparameter tuning.

## Features

- **GPU Monitoring**: Real-time monitoring of GPU usage to optimize resource utilization.
- **Data Augmentation**: Increases model robustness by applying random transformations to the training data.
- **Hyperparameter Tuning**: Utilizes Keras Tuner to find the optimal hyperparameters for the model.
- **Modular Architecture**: Each component of the project is modularized for better maintainability and scalability.

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- Keras Tuner
- CUDA and cuDNN (for GPU support)
- GPUtil

Install the required packages using pip:

```bash
pip install tensorflow keras-tuner GPUtil
```

## Setup

1. **Download the Dataset**: Place your image dataset in a directory named `training`. It should be organized in subdirectories, each named according to a class label.
   
2. **Configure Constants**: Update `config/constants.py` to suit your dataset and training preferences.

3. **Verify GPU Setup**: Ensure your system has the correct GPU drivers and CUDA/cuDNN versions installed. You can check GPU availability with:

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```

## Running the Project

Execute the main script to start the training and hyperparameter tuning process:

```bash
python main.py
```

The script will:

- Load and preprocess the data.
- Split the data into training, validation, and test sets.
- Perform hyperparameter tuning using Keras Tuner.
- Train the model with the best-found hyperparameters.
- Output GPU usage statistics during training.

## Customization

- **Model Architecture**: Modify `model/model.py` to experiment with different CNN architectures.
- **Hyperparameter Search Space**: Adjust `model/tuner.py` to explore different hyperparameter combinations.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.