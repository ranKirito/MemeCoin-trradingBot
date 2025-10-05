import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
from imblearn.combine import SMOTETomek  # Updated from NearMiss to SMOTE-Tomek

# Load the dataset
data = pd.read_csv("Data_30.csv")

# Define features and target
X = data.drop(columns=["target"])
y = data["target"]

# Handle missing values
X.fillna(0, inplace=True)

# Apply SMOTE-Tomek for balancing
smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train LightGBM model with best parameters
model = lgb.LGBMClassifier(
    learning_rate=0.01,
    max_depth=-1,
    n_estimators=50,
    num_leaves=31,
)
model.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Find optimal threshold (closest to the top-left corner of ROC curve)
gmeans = np.sqrt(tpr * (1 - fpr))
optimal_idx = np.argmax(gmeans)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold}")

# Apply optimal threshold to get class predictions
y_pred_opt = (y_probs >= optimal_threshold).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_probs)
report = classification_report(y_test, y_pred_opt)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(report)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.scatter(
    fpr[optimal_idx],
    tpr[optimal_idx],
    marker="o",
    color="red",
    label=f"Optimal Threshold ({optimal_threshold:.2f})",
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic Curve")
plt.legend()
plt.grid()
plt.show()
