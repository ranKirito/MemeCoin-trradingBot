from imblearn.under_sampling import NearMiss
import lightgbm as lgb
from utils import load_dataset, balanced_split, plot_and_save_feature_importance, save_feature_importance


def main():
    X, y = load_dataset("Data_20.csv", fill="median")
    X_train, X_test, y_train, y_test = balanced_split(
        X, y, resampler=NearMiss(), test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_jobs=-1,
    )
    # Use early stopping with the test split as eval set for quick utilities
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    feature_names = X.columns
    save_feature_importance(model, feature_names, "feature_importance.csv")
    plot_and_save_feature_importance(model, feature_names, "feature_importance.png")


if __name__ == "__main__":
    main()
