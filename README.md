# Chess Men Classification Project

This project focuses on classifying chess pieces using machine learning models, including MobileNetV2, InceptionV3, and DenseNet201.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Techniques Used](#techniquesUsed)
- [Models](#models)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

Chess Men Classification is a machine learning project aimed at recognizing and classifying chess pieces from images. The project utilizes three different deep learning models: MobileNetV2, InceptionV3, and DenseNet201, to achieve accurate classification.

## Dataset

The dataset used for training and testing the models can be found on Kaggle. The Chessman Image Dataset is available at:

[Chessman Image Dataset on Kaggle](https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset)

This dataset consists of labeled images of chess pieces. Each image is labeled with the corresponding chess piece (e.g., pawn, knight, bishop, rook, queen, king).


## Techniques Used

### Data Balancing

To address potential class imbalance in the dataset, we implemented data balancing techniques. This ensures that the machine learning models are not biased towards predicting the majority class and improves the overall model performance.

### Data Augmentation

Data augmentation is a key technique used to artificially increase the size of the dataset by applying various transformations to the existing images. Common augmentations include rotation, flipping, zooming, and changes in brightness. This helps improve the model's ability to generalize and handle variations in input data.

### Early Stopping

To prevent overfitting and find the optimal number of training epochs, we employed early stopping during model training. Early stopping monitors the model's performance on a validation set and halts training when the performance stops improving, preventing the model from learning noise in the training data.


## Models

1. **MobileNetV2**: MobileNetV2 is a lightweight deep learning model designed for mobile and edge devices. It balances accuracy and efficiency, making it suitable for real-time applications.

2. **InceptionV3**: InceptionV3 is a widely-used deep convolutional neural network architecture known for its performance on image classification tasks. It has a sophisticated architecture with multiple inception modules.

3. **DenseNet201**: DenseNet201 is a dense convolutional network that connects each layer to every other layer in a feed-forward fashion. It encourages feature reuse and helps mitigate vanishing gradient problems.

## Usage

In this example, users are instructed to open the Colab notebook and run a specific cell that installs the required libraries using the `!pip install` command. Adjust the dependencies in the command according to the libraries used in your project.

## Training

### Model Training Example (DenseNet201)

Below is an example of training the Chess Men Classification model using DenseNet201 as the base model:

```python
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Create the base DenseNet201 model
base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Create the model
model_DenseNet201 = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model_DenseNet201.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
history = model_DenseNet201.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

## Evaluation

Evaluate the performance of the trained models using the testing dataset. The evaluation script computes metrics such as accuracy, precision, recall, and F1 score. Run the evaluation script to obtain performance metrics and analyze the model's effectiveness.

Example command:

```bash
eval_result = model_DenseNet201.evaluate(validation_generator)
print("Validation Accuracy:", eval_result[1])

