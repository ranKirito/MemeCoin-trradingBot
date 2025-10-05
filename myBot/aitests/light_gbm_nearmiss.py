import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.under_sampling import NearMiss

# Load Data
df = pd.read_csv("Data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Handle missing values
X.fillna(X.median(), inplace=True)

# Apply NearMiss undersampling
nm = NearMiss()
X_resampled, y_resampled = nm.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Initialize LightGBM model
model = lgb.LGBMClassifier(
    boosting_type="gbdt", objective="binary", metric="binary_logloss", random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"ROC AUC Score: {roc_auc}")
print(classification_report(y_test, y_pred))
