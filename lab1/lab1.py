import pandas as pd

# Step 1: Load the data
data = pd.read_csv("bioresponse.csv")

# Step 2: Inspect the data to understand its structure
print(data.head())  # Display the first 5 rows of the dataset
print(data.info())  # Display information about data types and non-null counts

# Step 3: Split the data into features (X) and labels (y)
from sklearn.model_selection import train_test_split

X = data.drop(columns=["Activity"])  # Features
y = data["Activity"]  # Labels

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train four classifiers: shallow tree, deep tree, shallow forest, and deep forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the classifiers
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)  # Shallow decision tree
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)  # Deep decision tree
shallow_forest = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=42)  # Random forest with shallow trees
deep_forest = RandomForestClassifier(max_depth=None, n_estimators=100, random_state=42)  # Random forest with deep trees

# Train the classifiers on the training set
shallow_tree.fit(X_train, y_train)
deep_tree.fit(X_train, y_train)
shallow_forest.fit(X_train, y_train)
deep_forest.fit(X_train, y_train)

# Step 5: Evaluate the models using performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Function to evaluate a model and return performance metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)  # Predicted labels
    y_proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Log-Loss": log_loss(y_test, y_proba),
    }

# Evaluate each model and store the results in a dictionary
results = {
    "Shallow Tree": evaluate_model(shallow_tree, X_test, y_test),
    "Deep Tree": evaluate_model(deep_tree, X_test, y_test),
    "Shallow Forest": evaluate_model(shallow_forest, X_test, y_test),
    "Deep Forest": evaluate_model(deep_forest, X_test, y_test),
}

# Convert the results dictionary into a DataFrame for better visualization
results_df = pd.DataFrame(results).T
print(results_df)  # Display the performance metrics for all models

# Step 6: Plot Precision-Recall and ROC curves for each model
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

plt.figure(figsize=(12, 6))

# Precision-Recall Curve
plt.subplot(1, 2, 1)
for name, model in [("Shallow Tree", shallow_tree), ("Deep Tree", deep_tree), ("Shallow Forest", shallow_forest), ("Deep Forest", deep_forest)]:
    y_proba = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities
    precision, recall, _ = precision_recall_curve(y_test, y_proba)  # Compute Precision-Recall
    plt.plot(recall, precision, label=name)  # Plot the curve
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

# ROC Curve
plt.subplot(1, 2, 2)
for name, model in [("Shallow Tree", shallow_tree), ("Deep Tree", deep_tree), ("Shallow Forest", shallow_forest), ("Deep Forest", deep_forest)]:
    y_proba = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities
    fpr, tpr, _ = roc_curve(y_test, y_proba)  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")  # Plot ROC
plt.plot([0, 1], [0, 1], "k--", lw=0.7)  # Diagonal line for random predictions
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.tight_layout()
plt.show()

# Step 7: Adjust threshold to minimize Type II errors (False Negatives)
threshold = 0.3  # Set a lower threshold
y_proba = deep_forest.predict_proba(X_test)[:, 1]  # Predicted probabilities for positive class
y_pred_custom = (y_proba > threshold).astype(int)  # Convert probabilities to binary predictions

# Evaluate the custom threshold model
custom_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred_custom),
    "Precision": precision_score(y_test, y_pred_custom),
    "Recall": recall_score(y_test, y_pred_custom),
    "F1-Score": f1_score(y_test, y_pred_custom),
}

# Organize metrics into a DataFrame for better visualization
custom_metrics_df = pd.DataFrame([custom_metrics]).T.rename(columns={0: "Value"})
custom_metrics_df["Value"] = custom_metrics_df["Value"].apply(lambda x: f"{x:.4f}")  # Format to 4 decimal places

# Display the custom metrics
print(custom_metrics_df)
