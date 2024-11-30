# Import necessary libraries
from ucimlrepo import fetch_ucirepo  # To fetch datasets from the UCI ML repository
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import AdaBoostClassifier  # Adaboost model
from sklearn.model_selection import train_test_split, learning_curve  # Data splitting and learning curve visualization
from sklearn.metrics import accuracy_score, log_loss  # Metrics for model evaluation
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # Plotting

# ---------------------------------------------
# Step 1: Load the Banknote Authentication Dataset
# ---------------------------------------------
# Fetch the dataset using ucimlrepo
banknote_authentication = fetch_ucirepo(id=267)

# Split the data into features (X) and labels (y)
X = banknote_authentication.data.features  # Feature variables
y = banknote_authentication.data.targets   # Target variable (labels)

# ---------------------------------------------
# Step 2: Split data into training and testing sets
# ---------------------------------------------
# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------
# Step 3: Train Logistic Regression with Logistic Loss
# ---------------------------------------------
# Initialize the Logistic Regression model
logistic_model = LogisticRegression(solver='lbfgs', random_state=42)
logistic_model.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
y_pred_logistic = logistic_model.predict(X_test)  # Predicted classes
y_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1

# Evaluate the model's performance
logistic_accuracy = accuracy_score(y_test, y_pred_logistic)  # Accuracy of predictions
logistic_loss = log_loss(y_test, y_proba_logistic)  # Logistic Loss (Log-Loss)

# Print results for Logistic Regression
print(f"Logistic Loss - Accuracy: {logistic_accuracy:.4f}, Log-Loss: {logistic_loss:.4f}")

# ---------------------------------------------
# Step 4: Train Adaboost Classifier
# ---------------------------------------------
# Initialize the Adaboost model with 50 estimators
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)  # Train the model

# Make predictions on the test set
y_pred_adaboost = adaboost_model.predict(X_test)  # Predicted classes
y_proba_adaboost = adaboost_model.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1

# Evaluate the model's performance
adaboost_accuracy = accuracy_score(y_test, y_pred_adaboost)  # Accuracy of predictions

# Print results for Adaboost Classifier
print(f"Adaboost Loss - Accuracy: {adaboost_accuracy:.4f}")

# ---------------------------------------------
# Step 5: Calculate Binary Cross-Entropy Loss
# ---------------------------------------------
# Custom function to calculate Binary Cross-Entropy
def binary_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)  # Convert to numpy array
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0) by clipping probabilities
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Calculate Binary Cross-Entropy Loss manually
bce_loss = binary_crossentropy(y_test.to_numpy(), y_proba_logistic)
print(f"Binary Cross-Entropy - BCE Loss: {bce_loss:.4f}")

# ---------------------------------------------
# Step 6: Visualize Learning Curves
# ---------------------------------------------
# Function to plot learning curves
def plot_learning_curve(estimator, title, X_train, y_train):
    # Compute training and validation scores for varying sizes of training data
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_scores_mean = train_scores.mean(axis=1)  # Average training scores
    test_scores_mean = test_scores.mean(axis=1)  # Average validation scores

    # Plot the learning curves
    plt.figure()
    plt.title(title)  # Set the title of the plot
    plt.plot(train_sizes, train_scores_mean, label="Training score")  # Training curve
    plt.plot(train_sizes, test_scores_mean, label="Validation score")  # Validation curve
    plt.xlabel("Training examples")  # X-axis label
    plt.ylabel("Score")  # Y-axis label
    plt.legend(loc="best")  # Add legend
    plt.grid()  # Add grid for better visualization
    plt.show()

# Plot learning curve for Logistic Regression
plot_learning_curve(logistic_model, "Learning Curve - Logistic Loss", X_train, y_train)

# Plot learning curve for Adaboost
plot_learning_curve(adaboost_model, "Learning Curve - Adaboost Loss", X_train, y_train)

# Plot learning curve for Binary Cross-Entropy (Logistic Regression)
plot_learning_curve(logistic_model, "Learning Curve - Binary Cross-Entropy", X_train, y_train)

# ---------------------------------------------
# Step 7: Compare Results
# ---------------------------------------------
print("\nSummary of Results:")
print(f"Logistic Loss - Accuracy: {logistic_accuracy:.4f}, Log-Loss: {logistic_loss:.4f}")
print(f"Adaboost Loss - Accuracy: {adaboost_accuracy:.4f}")
print(f"Binary Cross-Entropy - BCE Loss: {bce_loss:.4f}")
