import pandas as pd
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from utils import load_dataset, balanced_split
from sklearn.metrics import classification_report, accuracy_score


def main():
    X, y = load_dataset("Data_20.csv", fill="median")
    X_train, X_test, y_train, y_test = balanced_split(
        X, y, resampler=NearMiss(), test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns).sort_values(
        ascending=False
    )
    print(feature_importance.head(20))


if __name__ == "__main__":
    main()
