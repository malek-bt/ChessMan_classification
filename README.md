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
- [Future Work](#futureWork)

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

```
## Evaluation

Evaluate the performance of the trained models using the testing dataset. The evaluation script computes metrics such as accuracy, precision, recall, and F1 score. Run the evaluation script to obtain performance metrics and analyze the model's effectiveness.

Example command:

```bash
eval_result = model_DenseNet201.evaluate(validation_generator)
print("Validation Accuracy:", eval_result[1])

```

## Results

### Model Performance

After evaluating the trained models, DenseNet201 demonstrated the highest accuracy among the models, achieving an accuracy of 84%. This indicates that DenseNet201 performed exceptionally well in classifying chess pieces on the validation set.


## Acknowledgments

This project has been made possible through the contributions and support of various individuals and organizations. We extend our sincere thanks to:

- **Kaggle Community:** For providing the Chessman Image Dataset used in this project. The Kaggle community has been instrumental in fostering a collaborative environment for data science and machine learning.

- **TensorFlow and Keras Developers:** We express our appreciation to the developers of TensorFlow and Keras for creating powerful deep learning frameworks that facilitated the implementation of complex models in this project.

- **Colab Notebooks:** The project extensively utilized Google Colab Notebooks for its ease of use and access to GPU resources. Colab greatly accelerated model training and experimentation.

- **Open Source Contributors:** Many open-source libraries and tools have played a crucial role in the development of this project. We are grateful for the efforts of the open-source community that continually contributes to the field of machine learning.


## Testing

To test the trained models on new images, follow the steps below:

1. **Add a New Model:**
   - If you have a new model you'd like to test, make sure the model is saved in a compatible format (e.g., Keras model in HDF5 format).

2. **Download or Prepare a New Image:**
   - Select or obtain an image that you want to use for testing. Ensure that the image is in a supported format (e.g., JPEG or PNG).

3. **Update the Test Code:**
   - Open the provided test script or code snippet 
   - Locate the section where the model is loaded, and update the model path to point to your new model.

   ```python
   # Load the saved model (update the model path)
   loaded_model = load_model('/path/to/your/new_model.h5')
   ```

## Future Work

While the current version of the Chess Men Classification project has achieved notable success, there are opportunities for further improvement and expansion. Future work may include:

- Fine-tuning hyperparameters for even better model performance.
- Exploring additional pre-trained models or architectures to compare and enhance classification accuracy.
- Implementing more advanced techniques, such as transfer learning from related domains or experimenting with ensemble methods.
- Collaborating with the community to gather additional labeled data or contributing to open-source initiatives in image classification.

We welcome contributions and ideas from the community to further advance this project.


