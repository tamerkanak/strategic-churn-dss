"""Model evaluation, threshold analysis, and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .constants import FALSE_NEGATIVE_COST, FALSE_POSITIVE_COST


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute standard churn classification metrics for a probability threshold."""

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "cost": int(fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST),
    }


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    false_negative_cost: int = FALSE_NEGATIVE_COST,
    false_positive_cost: int = FALSE_POSITIVE_COST,
) -> pd.DataFrame:
    """Evaluate F1 and business cost across possible classification thresholds."""

    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
        total = tn + fp + fn + tp
        rows.append(
            {
                "threshold": threshold,
                "accuracy": (tp + tn) / total,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "true_negative": int(tn),
                "business_cost": int(fn * false_negative_cost + fp * false_positive_cost),
            }
        )
    return pd.DataFrame(rows)


def best_thresholds(thresholds: pd.DataFrame) -> dict[str, float]:
    """Return F1-optimal, accuracy-optimal, and cost-minimizing thresholds."""

    f1_row = thresholds.sort_values(["f1", "recall"], ascending=[False, False]).iloc[0]
    accuracy_row = thresholds.assign(
        accuracy=(
            thresholds["true_positive"] + thresholds["true_negative"]
        )
        / (
            thresholds["true_positive"]
            + thresholds["true_negative"]
            + thresholds["false_positive"]
            + thresholds["false_negative"]
        )
    ).sort_values(["accuracy", "f1"], ascending=[False, False]).iloc[0]
    cost_row = thresholds.sort_values(["business_cost", "recall"], ascending=[True, False]).iloc[0]
    return {
        "f1_optimal_threshold": float(f1_row["threshold"]),
        "accuracy_optimal_threshold": float(accuracy_row["threshold"]),
        "cost_sensitive_threshold": float(cost_row["threshold"]),
        "minimum_business_cost": float(cost_row["business_cost"]),
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def plot_model_comparison(model_comparison: pd.DataFrame, path: Path) -> None:
    """Save a PR-AUC comparison chart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    ordered = model_comparison.sort_values("cv_pr_auc_mean", ascending=True)
    sns.barplot(data=ordered, x="cv_pr_auc_mean", y="model", color="#2f80ed")
    plt.xlabel("Cross-validated PR-AUC")
    plt.ylabel("")
    plt.title("Model comparison by PR-AUC")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_confusion(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float, path: Path) -> None:
    """Save confusion matrix chart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (probabilities >= threshold).astype(int)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        colorbar=False,
    )
    display.ax_.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_curves(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, roc_path: Path, pr_path: Path) -> None:
    """Save ROC and precision-recall curves."""

    roc_path.parent.mkdir(parents=True, exist_ok=True)
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("ROC curve")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Precision-recall curve")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=180)
    plt.close()


def plot_risk_distribution(decision_outputs: pd.DataFrame, path: Path) -> None:
    """Save risk tier distribution chart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    order = ["low", "medium", "high"]
    plt.figure(figsize=(7, 4.5))
    sns.countplot(data=decision_outputs, x="risk_tier", order=order, palette="viridis", hue="risk_tier", legend=False)
    plt.xlabel("Risk tier")
    plt.ylabel("Customers")
    plt.title("Operational churn risk distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_tenure_summary(tenure_df: pd.DataFrame, path: Path) -> None:
    """Save lifecycle churn chart."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=tenure_df, x="tenure_band", y="churn_rate", color="#27ae60")
    plt.xlabel("Tenure band")
    plt.ylabel("Observed churn rate")
    plt.title("Lifecycle-oriented churn by tenure band")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
