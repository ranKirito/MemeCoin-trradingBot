import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
)
from imblearn.combine import SMOTETomek  # Using SMOTE-Tomek for balancing
import joblib

# Load dataset
data = pd.read_csv("Data_30.csv")

# Define features and target
X = data.drop(columns=["target"])
y = data["target"]

# Handle missing values
X.fillna(0, inplace=True)

# Apply SMOTE-Tomek to balance the dataset
smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train LightGBM model with cost-sensitive learning via class_weight.
# A higher weight is given to class 0 (negative class) to reduce false positives.
model = lgb.LGBMClassifier(
    learning_rate=0.05,
    max_depth=-1,
    n_estimators=200,
    num_leaves=31,
    class_weight={0: 1.5, 1: 1},
)
model.fit(X_train, y_train)

# Get predicted probabilities for the positive class
y_probs = model.predict_proba(X_test)[:, 1]

# Apply the decision threshold (set at 0.55 based on our analysis)
optimal_threshold = 0.55
y_pred_opt = (y_probs >= optimal_threshold).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_probs)
report = classification_report(y_test, y_pred_opt)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(report)

# Save the current AI model to disk
joblib.dump(model, "coin_predictor_model_with_penalty30.pkl")
print("Model saved to 'coin_predictor_model_with_penalty30.pkl'")
