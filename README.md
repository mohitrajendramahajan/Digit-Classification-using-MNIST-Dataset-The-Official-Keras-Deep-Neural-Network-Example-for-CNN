# 📘 Digit Classification using CNN on MNIST

A complete deep‑learning pipeline using TensorFlow/Keras

This repository contains a full implementation of a Convolutional Neural Network (CNN) for handwritten digit classification using the MNIST dataset.

The project follows the official Keras example and expands it with preprocessing, visualization, model evaluation, confusion matrix, and training diagnostics.


## ✅ Project Overview

The goal of this project is to build a CNN that can classify handwritten digits (0–9) from the MNIST dataset.

The workflow includes:

- Setting reproducible seeds
  
- Loading and visualizing MNIST data
  
- Preprocessing (normalization, reshaping, one‑hot encoding)
  
- Designing a CNN with Conv2D → BatchNorm → ReLU → MaxPooling blocks
  
- Training the model with validation
  
- Evaluating performance on test data
  
- Generating predictions
  
- Plotting accuracy/loss curves
  
- Creating a confusion matrix
  
- Generating a classification report
  

## 📂 Dataset

### MNIST consists of:

- 60,000 training images
  
- 10,000 test images
  
- Grayscale images of size 28 × 28 × 1
  
- Labels from 0 to 9
  

## 🧪 Environment Setup

`conda install tensorflow`

`pip install numpy pandas matplotlib seaborn scikit-learn`



## 🧱 Model Architecture

The CNN follows this structure:

`Input (28×28×1)                                 `

`│                                               `

`├── Conv2D (8 filters, 3×3, same padding)       `

`├── BatchNormalization                          `

`├── ReLU                                        `

`├── MaxPooling2D (2×2)                          `

`│                                               `

`├── Conv2D (16 filters, 3×3, same padding)      `

`├── BatchNormalization                          `

`├── ReLU                                        `

`├── MaxPooling2D (2×2)                          `

`│                                               `

`├── Conv2D (32 filters, 3×3, same padding)      `

`├── BatchNormalization                          `

`├── ReLU                                        `

`│                                               `

`├── Flatten                                     `

`├── Dense (128 units, ReLU)                     `

`└── Dense (10 units, Softmax)                   `


This design extracts hierarchical features and reduces spatial dimensions while increasing depth.

## 🛠️ Code Workflow

1. Set Random Seeds

Ensures reproducibility for NumPy and TensorFlow.

2. Load MNIST Dataset

`(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()`


3. Visualize Sample Images
   
Displays the first 9 training images.

4. Preprocessing
 
- Normalize pixel values to [0,1]
  
- Reshape to (28,28,1)
  
- One‑hot encode labels using to_categorical
  
5. Build the CNN Model
  
Uses `Sequential()` with Conv2D, BatchNorm, ReLU, MaxPooling, Dense, and Softmax layers.

6. Compile the Model

`model.compile(optimizer=SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])`


7. Train the Model
   
Uses 20% of training data for validation.

8. Evaluate on Test Data
    
Outputs test accuracy and loss.

9. Predictions
    
Predicts the first 5 test images and compares with ground truth.

10. Training Curves
    
Plots accuracy and loss for both training and validation.

11. Confusion Matrix

Visualizes model performance across all classes.

12. Classification Report
    
Shows precision, recall, and F1‑score for each digit.


## 📊 Results 

### Typical performance:

- Training accuracy: ~99–100%
  
- Validation accuracy: ~98%
  
- Test accuracy: ~98%
  
- Confusion matrix shows strong performance across all digits.
  

## 📈 Visualizations Included

- Sample MNIST images
  
- Training vs validation accuracy
  
- Training vs validation loss
  
- Confusion matrix heatmap
  
- Classification report
  

## 📦 Files in This Repository

`├── mnist_cnn.ipynb / mnist_cnn.py   # Main code                          `

`├── model_plot.png                   # Model architecture visualization   `

`├── README.md                        # Project documentation              `



## 🚀 Future Improvements

- Add dropout layers to reduce overfitting
  
- Experiment with Adam optimizer
  
- Add data augmentation
  
- Try deeper architectures (LeNet‑5, VGG‑style)

## Kaggle link for reference:

**https://www.kaggle.com/code/mohitrajemdramahajan/digit-classification-using-mnist**
