"""SHAP and LIME explanation artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from .constants import ID_COLUMN


def _class_one_shap_values(raw_values: Any) -> np.ndarray:
    """Normalize SHAP output formats to a 2D class-one explanation matrix."""

    values = raw_values.values if hasattr(raw_values, "values") else raw_values
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[2] > 1:
            values = values[:, :, 1]
        else:
            values = values[:, :, 0]
    return values


def _feature_names(pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocessor"].get_feature_names_out())


def _transform(pipeline, x: pd.DataFrame) -> np.ndarray:
    return pipeline.named_steps["preprocessor"].transform(x)


def compute_shap_artifacts(
    pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    figure_dir: Path,
    max_rows: int = 500,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute global and local SHAP outputs for a fitted individual pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    feature_names = _feature_names(pipeline)
    model = pipeline.named_steps["model"]
    x_train_transformed = _transform(pipeline, x_train)
    x_test_transformed = _transform(pipeline, x_test)

    sample_size = min(max_rows, len(x_test_transformed))
    top_probability_positions = np.argsort(probabilities)[::-1][:sample_size]
    x_sample = x_test.iloc[top_probability_positions].copy()
    transformed_sample = x_test_transformed[top_probability_positions]
    sampled_probabilities = probabilities[top_probability_positions]

    rng = np.random.default_rng(random_state)
    background_size = min(300, len(x_train_transformed))
    background_positions = rng.choice(len(x_train_transformed), background_size, replace=False)
    background = x_train_transformed[background_positions]

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(transformed_sample)
    except Exception:
        explainer = shap.Explainer(model.predict_proba, background)
        raw_values = explainer(transformed_sample)

    shap_values = _class_one_shap_values(raw_values)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_global_importance.csv", index=False)

    top_features = importance.head(20).sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_features, x="mean_abs_shap", y="feature", color="#8e44ad")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    plt.title("Top global churn drivers")
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_global_importance.png", dpi=180)
    plt.close()

    local_rows: list[dict[str, Any]] = []
    for row_position, (_, original_row) in enumerate(x_sample.iterrows()):
        values = shap_values[row_position]
        driver_positions = np.argsort(np.abs(values))[::-1][:5]
        driver_labels = []
        for position in driver_positions:
            direction = "increases churn risk" if values[position] > 0 else "decreases churn risk"
            driver_labels.append(f"{feature_names[position]} ({direction}, {values[position]:.3f})")
        local_rows.append(
            {
                "customerID": original_row.get(ID_COLUMN, ""),
                "churn_probability": float(sampled_probabilities[row_position]),
                "top_shap_drivers": "; ".join(driver_labels),
            }
        )

    local_explanations = pd.DataFrame(local_rows)
    local_explanations.to_csv(output_dir / "shap_local_explanations.csv", index=False)
    return importance, local_explanations


def compute_lime_artifacts(
    pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    probabilities: np.ndarray,
    output_dir: Path,
    max_rows: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute a small set of LIME local explanations for stakeholder narratives."""

    from lime.lime_tabular import LimeTabularExplainer

    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = _feature_names(pipeline)
    model = pipeline.named_steps["model"]
    x_train_transformed = _transform(pipeline, x_train)
    x_test_transformed = _transform(pipeline, x_test)

    explainer = LimeTabularExplainer(
        training_data=x_train_transformed,
        feature_names=feature_names,
        class_names=["No churn", "Churn"],
        mode="classification",
        discretize_continuous=False,
        random_state=random_state,
    )

    top_probability_positions = np.argsort(probabilities)[::-1][: min(max_rows, len(x_test))]
    rows: list[dict[str, Any]] = []
    for position in top_probability_positions:
        explanation = explainer.explain_instance(
            x_test_transformed[position],
            model.predict_proba,
            num_features=5,
            labels=(1,),
        )
        rows.append(
            {
                "customerID": x_test.iloc[position].get(ID_COLUMN, ""),
                "churn_probability": float(probabilities[position]),
                "lime_explanation": "; ".join(
                    f"{feature}: {weight:.3f}" for feature, weight in explanation.as_list(label=1)
                ),
            }
        )

    lime_df = pd.DataFrame(rows)
    lime_df.to_csv(output_dir / "lime_local_explanations.csv", index=False)
    return lime_df
