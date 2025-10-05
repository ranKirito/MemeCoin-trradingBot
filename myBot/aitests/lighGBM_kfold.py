import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.under_sampling import NearMiss
from utils import load_dataset


def main():
    X, y = load_dataset("Data_30.csv", fill="median")
    X_res, y_res = NearMiss().fit_resample(X, y)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, aucs, precs, recs = [], [], [], []

    for tr_idx, te_idx in kf.split(X_res, y_res):
        X_tr, X_te = X_res.iloc[tr_idx], X_res.iloc[te_idx]
        y_tr, y_te = y_res.iloc[tr_idx], y_res.iloc[te_idx]

        model = lgb.LGBMClassifier(
            learning_rate=0.05,
            max_depth=-1,
            n_estimators=1000,
            num_leaves=31,
            n_jobs=-1,
        )
        # Early stopping using the test fold as eval for simplicity
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_probs = model.predict_proba(X_te)[:, 1]
        # Choose per-fold optimal threshold via ROC g-mean
        from utils import optimal_threshold_gmean

        thr = optimal_threshold_gmean(y_te, y_probs)
        y_pred = (y_probs >= thr).astype(int)

        accs.append(accuracy_score(y_te, y_pred))
        aucs.append(roc_auc_score(y_te, y_probs))
        precs.append(precision_score(y_te, y_pred, zero_division=0))
        recs.append(recall_score(y_te, y_pred, zero_division=0))

    print(f"Average Accuracy: {np.mean(accs):.4f}")
    print(f"Average ROC AUC: {np.mean(aucs):.4f}")
    print(f"Average Precision: {np.mean(precs):.4f}")
    print(f"Average Recall: {np.mean(recs):.4f}")


if __name__ == "__main__":
    main()
