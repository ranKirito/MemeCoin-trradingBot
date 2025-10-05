import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    roc_curve,
)


def load_dataset(path: str, target: str = "target", fill: str = "zero") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV, split into X/y, and fill missing values.
    fill: "zero" | "median"
    """
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    X = df.drop(columns=[target])
    y = df[target]
    if fill == "median":
        X = X.fillna(X.median(numeric_only=True))
    else:
        X = X.fillna(0)
    return X, y


def balanced_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    resampler=None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Optionally resample, then stratified split.
    """
    if resampler is not None:
        X, y = resampler.fit_resample(X, y)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate_from_probs(y_true: pd.Series, y_probs: np.ndarray, threshold: float) -> Dict[str, float | str]:
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "report": classification_report(y_true, y_pred),
    }


def optimal_threshold_gmean(y_true: pd.Series, y_probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = int(np.argmax(gmeans))
    return float(thresholds[idx])


def save_feature_importance(model, feature_names, path_csv: str, top_k: int | None = None) -> None:
    import pandas as pd

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False)
    if top_k is not None:
        fi = fi.head(top_k)
    fi.to_csv(path_csv, index=False)


def plot_and_save_feature_importance(model, feature_names, out_path: str) -> None:
    import matplotlib.pyplot as plt

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return
    order = np.argsort(importances)
    plt.figure(figsize=(8, max(4, len(feature_names) * 0.2)))
    plt.barh(np.array(feature_names)[order], np.array(importances)[order], color="steelblue")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()


def plot_and_save_roc(y_true: pd.Series, y_probs: np.ndarray, out_path: str, mark_optimal: bool = True) -> float:
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    idx = int(np.argmax(gmeans))
    thr = float(thresholds[idx])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    if mark_optimal:
        plt.scatter(fpr[idx], tpr[idx], color="red", label=f"Optimal thr={thr:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=144)
    plt.close()
    return thr

