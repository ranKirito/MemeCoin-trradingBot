import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.combine import SMOTETomek

# Load dataset
data = pd.read_csv("Data_30.csv")

# Define features and target
X = data.drop(columns=["target"])
y = data["target"]

# Handle missing values
X.fillna(0, inplace=True)

# Apply SMOTE-Tomek for balancing
smotetomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smotetomek.fit_resample(X, y)

# Set up 5-fold Stratified Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
roc_auc_scores = []
precision_scores = []
recall_scores = []

# Train and evaluate model using k-fold CV
for train_index, test_index in kf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    model = lgb.LGBMClassifier(
        learning_rate=0.01, max_depth=-1, n_estimators=50, num_leaves=31
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[
        :, 1
    ]  # Get probabilities for the positive class

    # Apply optimal threshold from ROC curve (0.49)
    optimal_threshold = 0.49
    y_pred_opt = (y_probs >= optimal_threshold).astype(int)

    # Evaluate performance
    accuracy_scores.append(accuracy_score(y_test, y_pred_opt))
    roc_auc_scores.append(roc_auc_score(y_test, y_probs))
    precision_scores.append(precision_score(y_test, y_pred_opt))
    recall_scores.append(recall_score(y_test, y_pred_opt))

# Print average scores across folds
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average ROC AUC Score: {np.mean(roc_auc_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
