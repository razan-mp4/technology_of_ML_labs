# Computer Practicum 1: Binary Classification with Decision Trees and Random Forests

## Overview

This repository contains the implementation of a computer practicum focused on binary classification using decision trees and random forests. The goal is to predict the biological activity of molecules based on a dataset of molecular features.

The practicum involves training, evaluating, and analyzing the performance of various machine learning models using Python and popular data science libraries. Additionally, we explore techniques for adjusting model thresholds to prioritize specific types of errors.

---

## Objectives

1. **Train Four Classifiers:**
   - Shallow Decision Tree
   - Deep Decision Tree
   - Random Forest with Shallow Trees
   - Random Forest with Deep Trees

2. **Evaluate Models Using Key Metrics:**
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Log-Loss

3. **Analyze Model Performance:**
   - Plot Precision-Recall and ROC curves.
   - Calculate AUC (Area Under the Curve) for ROC.

4. **Optimize for Specific Error Types:**
   - Adjust model thresholds to minimize Type II errors (False Negatives) and evaluate the resulting model.

---

## Dataset

The dataset used in this practicum, `bioresponse.csv`, contains molecular features (columns `D1` to `D1776`) and a target variable `Activity`, indicating the biological response (1 for active, 0 for inactive).

---

## Steps

### 1. Data Preparation
- Load the dataset and inspect its structure.
- Split the data into training and testing sets.

### 2. Model Training
- Train decision tree and random forest models with different depths and configurations.

### 3. Model Evaluation
- Calculate performance metrics:
  - **Accuracy:** Proportion of correctly classified instances.
  - **Precision:** Proportion of true positives among predicted positives.
  - **Recall:** Proportion of true positives among actual positives.
  - **F1-Score:** Harmonic mean of precision and recall.
  - **Log-Loss:** Logarithmic loss function for probabilistic predictions.
- Compare the performance of all models.

### 4. Visualization
- Plot Precision-Recall and ROC curves for all models.
- Compute and display AUC values for the ROC curves.

### 5. Threshold Adjustment
- Adjust the classification threshold of the best-performing model to minimize Type II errors.
- Re-evaluate the model's performance using the adjusted threshold.

---

## Results

The results include:
1. Performance metrics for all models.
2. Precision-Recall and ROC visualizations.
3. Optimized model for minimizing Type II errors with updated metrics.

---

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - `pandas` for data manipulation
  - `scikit-learn` for machine learning models and metrics
  - `matplotlib` for visualizations

---
