import lightgbm as lgb
from imblearn.under_sampling import NearMiss
from utils import load_dataset, balanced_split, plot_and_save_roc, evaluate_from_probs


def main():
    X, y = load_dataset("Data_30.csv", fill="median")
    X_train, X_test, y_train, y_test = balanced_split(
        X, y, resampler=NearMiss(), test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        max_depth=-1,
        n_estimators=1000,
        num_leaves=31,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    y_probs = model.predict_proba(X_test)[:, 1]
    optimal_threshold = plot_and_save_roc(y_test, y_probs, "roc_curve.png", mark_optimal=True)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    metrics = evaluate_from_probs(y_test, y_probs, optimal_threshold)
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"ROC AUC Score: {metrics['roc_auc']}")
    print(metrics["report"])


if __name__ == "__main__":
    main()
