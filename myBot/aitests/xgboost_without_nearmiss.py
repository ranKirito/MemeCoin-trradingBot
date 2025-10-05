import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# Load the dataset
data = pd.read_csv("./Data.csv")

# Define features and target variable
X = data.drop(columns=["target"])
y = data["target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define XGBoost model with optimized parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=100,  # Reduce boosting rounds
    max_depth=5,  # Limit tree complexity
    learning_rate=0.1,
    subsample=0.8,  # Use a fraction of data per boosting round
    colsample_bytree=0.8,  # Use a fraction of features per tree
    tree_method="hist",  # Faster training method (change to 'gpu_hist' if GPU is available)
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(report)
