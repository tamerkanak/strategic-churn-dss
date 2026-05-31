"""End-to-end model training and artifact generation."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .constants import (
    ARTIFACT_DIR,
    CV_FOLDS,
    ID_COLUMN,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from .data import add_domain_features, clean_dataset, load_dataset, prepare_dataset, tenure_summary
from .evaluation import (
    best_thresholds,
    classification_metrics,
    plot_confusion,
    plot_curves,
    plot_model_comparison,
    plot_risk_distribution,
    plot_tenure_summary,
    save_json,
    threshold_analysis,
)
from .explainability import compute_lime_artifacts, compute_shap_artifacts
from .models import (
    ModelSpec,
    build_focused_voting_ensemble,
    build_model_pipeline,
    build_soft_voting_ensemble,
    build_stacking_ensemble,
    candidate_model_specs,
)
from .recommendations import add_decision_outputs


def _predict_proba(estimator: object, x: pd.DataFrame) -> np.ndarray:
    return estimator.predict_proba(x)[:, 1]


def _scoring() -> dict[str, str]:
    # String scorers keep cross-validation output stable across sklearn releases.
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }


def _mean_std(cv_results: dict[str, np.ndarray], metric: str) -> tuple[float, float]:
    values = cv_results[f"test_{metric}"]
    return float(np.mean(values)), float(np.std(values))


def _model_row(
    model_name: str,
    cv_results: dict[str, np.ndarray],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model_name}
    for metric in _scoring():
        mean, std = _mean_std(cv_results, metric)
        row[f"cv_{metric}_mean"] = mean
        row[f"cv_{metric}_std"] = std
    row.update({f"test_{key}": value for key, value in test_metrics.items()})
    return row


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _choose_explainer_model(
    model_comparison: pd.DataFrame,
    fitted_models: dict[str, object],
) -> tuple[str, object]:
    """Choose a fitted individual model that SHAP can explain reliably."""

    tree_model_tokens = ("CatBoost", "LightGBM", "XGBoost", "Random Forest", "Extra Trees", "Gradient")
    ordered_names = model_comparison.sort_values("cv_pr_auc_mean", ascending=False)["model"].tolist()
    for name in ordered_names:
        if any(token in name for token in tree_model_tokens) and name in fitted_models:
            return name, fitted_models[name]
    for name in ordered_names:
        if name in fitted_models:
            return name, fitted_models[name]
    raise RuntimeError("No fitted individual model is available for explanation.")


def _select_model_name(model_comparison: pd.DataFrame, pr_auc_tolerance: float = 0.003) -> str:
    """Select a strong operational model without overreacting to tiny PR-AUC gaps."""

    top_pr_auc = float(model_comparison["cv_pr_auc_mean"].max())
    near_best = model_comparison[
        model_comparison["cv_pr_auc_mean"] >= top_pr_auc - pr_auc_tolerance
    ].copy()
    selected = near_best.sort_values(
        ["test_accuracy", "cv_accuracy_mean", "cv_f1_mean", "cv_pr_auc_mean"],
        ascending=[False, False, False, False],
    ).iloc[0]
    return str(selected["model"])


def run_training(
    data_source: str = "auto",
    random_state: int = RANDOM_STATE,
    artifact_dir: Path = ARTIFACT_DIR,
    report_dir: Path = REPORT_DIR,
    quick: bool = False,
) -> dict[str, Any]:
    """Train all models, select the operational model, and write deliverable artifacts."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    x, y, _customer_ids, source = prepare_dataset(data_source=data_source)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=random_state,
    )

    raw_df, _ = load_dataset(data_source=data_source)
    clean_df = clean_dataset(raw_df)
    featured_df = add_domain_features(clean_df)
    lifecycle = tenure_summary(featured_df)
    lifecycle.to_csv(artifact_dir / "tenure_summary.csv", index=False)
    plot_tenure_summary(lifecycle, figure_dir / "tenure_churn.png")

    dataset_summary = {
        "source": source,
        "rows": int(len(featured_df)),
        "columns_after_feature_engineering": int(featured_df.shape[1]),
        "raw_columns": int(raw_df.shape[1]),
        "churn_rate": float(y.mean()),
        "non_churn_customers": int((y == 0).sum()),
        "churn_customers": int((y == 1).sum()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    save_json(dataset_summary, artifact_dir / "dataset_summary.json")

    specs = candidate_model_specs(random_state)
    if quick:
        specs = [
            spec
            for spec in specs
            if spec.name in {"Logistic Regression (No SMOTE)", "Random Forest (No SMOTE Tuned)"}
        ]

    cv = StratifiedKFold(n_splits=CV_FOLDS if not quick else 3, shuffle=True, random_state=random_state)
    comparison_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, object] = {}
    successful_specs: dict[str, ModelSpec] = {}

    for spec in specs:
        print(f"Training and validating {spec.name}...")
        pipeline = build_model_pipeline(
            spec.estimator,
            x_train,
            random_state,
            sampler=spec.sampler,
            encoder=spec.encoder,
            scale_numeric=spec.scale_numeric,
            drop_columns=spec.drop_columns,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring=_scoring(),
                n_jobs=1,
                error_score="raise",
            )
            pipeline.fit(x_train, y_train)
        probabilities = _predict_proba(pipeline, x_test)
        metrics = classification_metrics(y_test, probabilities, threshold=0.50)
        comparison_rows.append(_model_row(spec.name, cv_results, metrics))
        fitted_models[spec.name] = pipeline
        successful_specs[spec.name] = spec

    individual_comparison = pd.DataFrame(comparison_rows)
    top_three_names = (
        individual_comparison.sort_values("cv_pr_auc_mean", ascending=False)["model"].head(3).tolist()
    )
    top_specs = [successful_specs[name] for name in top_three_names]

    ensemble = build_soft_voting_ensemble(top_specs, x_train, random_state)
    print(f"Training soft-voting ensemble from: {', '.join(top_three_names)}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_cv_results = cross_validate(
            ensemble,
            x_train,
            y_train,
            cv=cv,
            scoring=_scoring(),
            n_jobs=1,
            error_score="raise",
        )
        ensemble.fit(x_train, y_train)
    ensemble_probabilities = _predict_proba(ensemble, x_test)
    ensemble_metrics = classification_metrics(y_test, ensemble_probabilities, threshold=0.50)
    comparison_rows.append(_model_row("Soft Voting Ensemble", ensemble_cv_results, ensemble_metrics))

    stacking = build_stacking_ensemble(top_specs, x_train, random_state)
    print(f"Training stacking ensemble from: {', '.join(top_three_names)}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stacking_cv_results = cross_validate(
            stacking,
            x_train,
            y_train,
            cv=cv,
            scoring=_scoring(),
            n_jobs=1,
            error_score="raise",
        )
        stacking.fit(x_train, y_train)
    stacking_probabilities = _predict_proba(stacking, x_test)
    stacking_metrics = classification_metrics(y_test, stacking_probabilities, threshold=0.50)
    comparison_rows.append(_model_row("Stacking Ensemble", stacking_cv_results, stacking_metrics))

    focused_voting = build_focused_voting_ensemble(x_train, random_state)
    print("Training focused voting ensemble from iterative score search...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        focused_cv_results = cross_validate(
            focused_voting,
            x_train,
            y_train,
            cv=cv,
            scoring=_scoring(),
            n_jobs=1,
            error_score="raise",
        )
        focused_voting.fit(x_train, y_train)
    focused_probabilities = _predict_proba(focused_voting, x_test)
    focused_metrics = classification_metrics(y_test, focused_probabilities, threshold=0.50)
    comparison_rows.append(_model_row("Focused Voting Ensemble", focused_cv_results, focused_metrics))

    model_comparison = pd.DataFrame(comparison_rows).sort_values("cv_pr_auc_mean", ascending=False)
    model_comparison.to_csv(artifact_dir / "model_comparison.csv", index=False)
    plot_model_comparison(model_comparison, figure_dir / "model_comparison.png")

    all_models = dict(fitted_models)
    all_models["Soft Voting Ensemble"] = ensemble
    all_models["Stacking Ensemble"] = stacking
    all_models["Focused Voting Ensemble"] = focused_voting
    best_model_name = _select_model_name(model_comparison)
    best_model = all_models[best_model_name]
    best_probabilities = _predict_proba(best_model, x_test)

    thresholds = threshold_analysis(y_test, best_probabilities)
    thresholds.to_csv(artifact_dir / "threshold_analysis.csv", index=False)
    threshold_summary = best_thresholds(thresholds)
    operational_threshold = threshold_summary["cost_sensitive_threshold"]

    default_metrics = classification_metrics(y_test, best_probabilities, threshold=0.50)
    accuracy_metrics = classification_metrics(
        y_test,
        best_probabilities,
        threshold=threshold_summary["accuracy_optimal_threshold"],
    )
    operational_metrics = classification_metrics(y_test, best_probabilities, threshold=operational_threshold)
    final_metrics = {
        "selected_model": best_model_name,
        "selection_policy": (
            "Primary ranking uses cross-validated PR-AUC. Models within 0.003 PR-AUC of "
            "the best candidate are treated as a practical tie, then ranked by "
            "hold-out accuracy, cross-validated accuracy, F1, and PR-AUC."
        ),
        "top_three_for_ensemble": top_three_names,
        "default_threshold_metrics": _serializable_metrics(default_metrics),
        "accuracy_optimal_threshold_metrics": _serializable_metrics(accuracy_metrics),
        "operational_threshold_metrics": _serializable_metrics(operational_metrics),
        "threshold_summary": threshold_summary,
    }
    save_json(final_metrics, artifact_dir / "final_metrics.json")

    test_customers = x_test.copy()
    decision_outputs = add_decision_outputs(
        test_customers,
        best_probabilities,
        threshold=operational_threshold,
    )
    decision_outputs[TARGET_COLUMN] = y_test.to_numpy()
    decision_outputs.to_csv(artifact_dir / "decision_outputs.csv", index=False)

    plot_confusion(y_test, best_probabilities, operational_threshold, figure_dir / "confusion_matrix.png")
    plot_curves(
        y_test,
        best_probabilities,
        figure_dir / "roc_curve.png",
        figure_dir / "precision_recall_curve.png",
    )
    plot_risk_distribution(decision_outputs, figure_dir / "risk_distribution.png")

    joblib.dump(best_model, artifact_dir / "churn_model.joblib")
    joblib.dump(
        {
            "model_name": best_model_name,
            "operational_threshold": operational_threshold,
            "features": [column for column in x.columns if column != ID_COLUMN],
        },
        artifact_dir / "model_metadata.joblib",
    )

    explainer_model_name, explainer_pipeline = _choose_explainer_model(model_comparison, fitted_models)
    explainer_probabilities = _predict_proba(explainer_pipeline, x_test)
    shap_importance, shap_local = compute_shap_artifacts(
        explainer_pipeline,
        x_train,
        x_test,
        explainer_probabilities,
        artifact_dir,
        figure_dir,
        random_state=random_state,
    )
    lime_local = compute_lime_artifacts(
        explainer_pipeline,
        x_train,
        x_test,
        explainer_probabilities,
        artifact_dir,
        random_state=random_state,
    )

    metadata = {
        "dataset_summary": dataset_summary,
        "selected_model": best_model_name,
        "explainer_model": explainer_model_name,
        "operational_threshold": operational_threshold,
        "artifact_dir": str(artifact_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "figure_dir": str(figure_dir.resolve()),
        "shap_rows": int(len(shap_local)),
        "lime_rows": int(len(lime_local)),
        "top_shap_features": shap_importance.head(10)["feature"].tolist(),
    }
    save_json(metadata, artifact_dir / "run_metadata.json")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the strategic churn DSS models.")
    parser.add_argument("--data-source", default="auto", help="auto, local, url, or a direct CSV path.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--quick", action="store_true", help="Run a reduced model set for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_training(
        data_source=args.data_source,
        random_state=args.random_state,
        quick=args.quick,
    )
    print("Training complete.")
    print(f"Selected model: {metadata['selected_model']}")
    print(f"Operational threshold: {metadata['operational_threshold']:.2f}")
    print(f"Artifacts: {metadata['artifact_dir']}")


if __name__ == "__main__":
    main()
