import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    fbeta_score,
    recall_score,
    precision_score,
    classification_report,
)
import lightgbm as lgb
from imblearn.combine import SMOTETomek  # Updated from NearMiss to SMOTE-Tomek


# Custom scoring function for grid search.
# Here we use a fixed threshold (0.66) solely for tuning.
def fbeta_at_threshold(estimator, X, y):
    y_probs = estimator.predict_proba(X)[:, 1]
    fixed_threshold = 0.66
    y_pred = (y_probs >= fixed_threshold).astype(int)
    return fbeta_score(y, y_pred, beta=1)


# Load data
df = pd.read_csv("Data_30.csv")  # Updated dataset with selected features
df = df.fillna(0)  # Replace NaN values with 0

# Define features and target
X = df.drop(columns=["target"])
y = df["target"]

# Apply SMOTE-Tomek for balancing
smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Define parameter grid for hyperparameter tuning
param_grid = {
    "num_leaves": [20, 31, 40],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200],
    "max_depth": [-1, 10, 20],
}

# Initialize LightGBM classifier with force_row_wise to remove overhead
lgb_clf = lgb.LGBMClassifier(force_row_wise=True)

# Run grid search using the custom F-beta scorer (with fixed threshold 0.66)
grid_search = GridSearchCV(
    lgb_clf, param_grid, cv=5, scoring=fbeta_at_threshold, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Get predicted probabilities for the positive class on the test set
y_probs = best_model.predict_proba(X_test)[:, 1]

# Define thresholds to try (in decimal format)
thresholds = [0.45, 0.49, 0.55, 0.60]
results = {}

# Evaluate performance at each threshold
for thr in thresholds:
    y_pred = (y_probs >= thr).astype(int)
    results[thr] = {
        "fbeta": fbeta_score(y_test, y_pred, beta=0.5),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

# Find the threshold with the best F-beta score
best_thr = max(results, key=lambda thr: results[thr]["fbeta"])
best_results = results[best_thr]

# Print best threshold and its metrics along with the best hyperparameters
print("Best Threshold:", best_thr)
print("Best Hyperparameters:", grid_search.best_params_)
print("Results for best threshold:")
print("F-beta Score (beta=0.5):", best_results["fbeta"])
print("Accuracy:", best_results["accuracy"])
print("ROC AUC Score:", best_results["roc_auc"])
print("Precision:", best_results["precision"])
print("Recall:", best_results["recall"])
print("\nClassification Report:\n", best_results["classification_report"])
