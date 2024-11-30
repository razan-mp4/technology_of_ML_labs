# Computer Practicum 2: Binary Classification with Logistic Regression and Adaboost

## Overview

This repository contains the implementation of a computer practicum focused on binary classification using logistic regression and Adaboost. The goal is to classify banknotes as authentic or fake based on extracted features from images of the banknotes.

The practicum involves training, evaluating, and analyzing the performance of machine learning models using Python and popular data science libraries. Additionally, we explore key loss functions such as Logistic Loss, Adaboost Loss, and Binary Cross-Entropy.

---

## Objectives

1. **Train and Evaluate Models Using Different Loss Functions:**
   - Logistic Regression with Logistic Loss
   - Adaboost with Exponential Loss
   - Logistic Regression with Binary Cross-Entropy Loss (calculated manually)

2. **Visualize Model Performance:**
   - Plot learning curves for training and validation sets for each model.

3. **Compare Models:**
   - Evaluate models using key metrics such as Accuracy and Log-Loss.
   - Analyze and interpret Binary Cross-Entropy.

---

## Dataset

The dataset used in this practicum is the **Banknote Authentication Dataset** from the UCI Machine Learning Repository. It contains four features extracted from images of banknotes and a target variable indicating whether the banknote is authentic (1) or fake (0).

---

## Steps

### 1. Data Preparation
- Load the dataset using `ucimlrepo`.
- Split the dataset into training and testing sets (80% training, 20% testing).

### 2. Model Training
- Train the following models:
  - Logistic Regression with Logistic Loss.
  - Adaboost Classifier.
  - Logistic Regression with Binary Cross-Entropy Loss (manually calculated).

### 3. Model Evaluation
- Compute and compare the following metrics:
  - **Accuracy:** Proportion of correctly classified instances.
  - **Log-Loss:** Measures the performance of probabilistic predictions.
  - **Binary Cross-Entropy (BCE):** Manually calculated to evaluate prediction probabilities.

### 4. Visualization
- Plot learning curves for:
  - Logistic Loss (Logistic Regression)
  - Adaboost Loss
  - Binary Cross-Entropy (Logistic Regression)

---

## Results

The results include:
1. **Performance Metrics:**
   - Accuracy, Log-Loss, and Binary Cross-Entropy for all models.
2. **Learning Curves:**
   - Visualization of training and validation performance for each model.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `ucimlrepo` for dataset fetching
  - `scikit-learn` for machine learning models and evaluation metrics
  - `numpy` for numerical computations
  - `matplotlib` for data visualization

---

## Conclusion

This practicum demonstrates the application of different loss functions in binary classification. Logistic Regression with Logistic Loss and Binary Cross-Entropy showed strong performance, while Adaboost demonstrated its ensemble-based strength for classification tasks. The visualizations provided insights into the learning behavior of each model.
