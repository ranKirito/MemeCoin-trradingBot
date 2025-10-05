import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.under_sampling import NearMiss

# Load data
data = pd.read_csv("./Data_20.csv")

# Drop non-numeric and irrelevant columns
data = data.select_dtypes(include=[np.number]).dropna()

# Separate features and target
X = data.drop(columns=["target"])  # Replace 'target' with the actual target column name
y = data["target"]

# Handle missing values by filling with the median value of each column
X = X.fillna(X.median())

# Apply NearMiss for undersampling
nm = NearMiss()
X_resampled, y_resampled = nm.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(report)
