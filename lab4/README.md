# Computer Practicum 4: Image Classification and Transfer Learning with Neural Networks

## Overview

This repository contains the implementation of a computer practicum focused on image classification using different neural network architectures. The task involves classifying images of cats and dogs using fully connected neural networks, convolutional neural networks (CNNs), and transfer learning with pre-trained models (VGG19 and ResNet50). The practicum demonstrates the concepts of overfitting and transfer learning while comparing the performance of different models.

---

## Objectives

1. **Implement Fully Connected Neural Networks (FCN):**
   - Design a fully connected network with three hidden layers for binary classification.
   - Train the model on images reshaped into one-dimensional vectors.

2. **Implement Convolutional Neural Networks (CNN):**
   - Design a CNN with two convolutional layers followed by max pooling layers for feature extraction.

3. **Apply Transfer Learning:**
   - Use pre-trained VGG19 and ResNet50 models, freezing their convolutional layers and training custom classification layers on the new dataset.

4. **Demonstrate Overfitting:**
   - Train the FCN and CNN models for an extended number of epochs to showcase overfitting.

5. **Evaluate and Compare Models:**
   - Evaluate the models on the test dataset and compare their performance in terms of accuracy.

---

## Dataset

The dataset used for this practicum consists of images of cats and dogs, organized into three folders:
1. **Train:** Training data with subfolders `cats` and `dogs`.
2. **Validation:** Validation data with subfolders `cats` and `dogs`.
3. **Test:** Test data with subfolders `cats` and `dogs`.

Each image is resized to 128x128 pixels for uniformity.

---

## Steps

### 1. Data Preprocessing
- Load images using Keras' `ImageDataGenerator` to:
  - Rescale pixel values between 0 and 1.
  - Load batches of images for training, validation, and testing.

### 2. Fully Connected Neural Network (FCN)
- Flatten the 3D images into 1D vectors.
- Use three dense layers with ReLU activation and a final sigmoid layer for binary classification.

### 3. Convolutional Neural Network (CNN)
- Add convolutional layers with ReLU activation followed by max pooling layers.
- Use dense layers and a sigmoid output layer for classification.

### 4. Transfer Learning with VGG19 and ResNet50
- Load the pre-trained VGG19 and ResNet50 models from Keras.
- Freeze all convolutional layers and add custom dense layers for classification.
- Train only the added layers on the dataset.

### 5. Overfitting Demonstration
- Increase the number of training epochs for the FCN and CNN to demonstrate overfitting:
  - Compare training and validation accuracy to observe performance degradation on the validation set.

### 6. Evaluation and Comparison
- Evaluate all models on the test dataset and compare their accuracies.

---

## Results

1. **Model Test Accuracies:**
   - Fully Connected Network: 50%
   - Convolutional Neural Network: 79%
   - Transfer Learning (VGG19): 86%
   - Transfer Learning (ResNet50): 67%

2. **Learning Curves:**
   - Training and validation accuracy curves for all models.
   - Overfitting observed in FCN and CNN with increased epochs.

3. **Performance Insights:**
   - Transfer learning with VGG19 outperformed other models due to pre-trained feature extraction.
   - CNN showed robust performance, leveraging spatial relationships in images.
   - FCN performed poorly due to the loss of spatial information when flattening images.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `tensorflow` for building and training neural networks
  - `matplotlib` for visualizing training curves
  - `keras` for pre-trained models and data preprocessing
  - `numpy` for numerical operations

---

## Conclusion

This practicum illustrates the strengths and limitations of different neural network architectures for image classification. The results highlight the benefits of convolutional layers for capturing spatial features and the power of transfer learning for improving performance on small datasets. Overfitting was effectively demonstrated, emphasizing the need for proper regularization and early stopping in real-world applications.

---
