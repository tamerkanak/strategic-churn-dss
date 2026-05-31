"""Model definitions for churn prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .features import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: object
    sampler: Literal["smote"] | None = None
    encoder: Literal["onehot", "target"] = "onehot"
    scale_numeric: bool = True
    drop_columns: tuple[str, ...] = ()


FOCUSED_ACCURACY_DROP_COLUMNS = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "StreamingTV",
    "StreamingMovies",
    "is_senior_m2m",
    "dependents_partner_score",
)

DROP_SPARSE_INTERACTION_COLUMNS = (
    "gender",
    "PhoneService",
    "MultipleLines",
    "monthly_x_tenure",
    "tenure_inverse",
    "tenure_charge_band",
    "payment_contract_group",
    "internet_service_group",
)

NO_DEMOGRAPHIC_DROP_COLUMNS = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "is_senior_m2m",
    "dependents_partner_score",
)


def candidate_model_specs(random_state: int) -> list[ModelSpec]:
    """Return the planned interpretable, nonlinear, boosting, and SVM candidates."""

    return [
        ModelSpec(
            "Logistic Regression (SMOTE)",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "Logistic Regression (No SMOTE)",
            LogisticRegression(
                max_iter=1500,
                solver="liblinear",
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "Logistic Regression (Target Encoded)",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=random_state,
            ),
            encoder="target",
        ),
        ModelSpec(
            "Logistic Regression (Focused Accuracy)",
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                random_state=random_state,
            ),
            drop_columns=FOCUSED_ACCURACY_DROP_COLUMNS,
        ),
        ModelSpec(
            "Random Forest (SMOTE)",
            RandomForestClassifier(
                n_estimators=220,
                max_depth=9,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "Random Forest (No SMOTE Tuned)",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=7,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "Extra Trees (No SMOTE)",
            ExtraTreesClassifier(
                n_estimators=800,
                max_depth=8,
                min_samples_leaf=8,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "SVM RBF (SMOTE)",
            SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "SVM RBF (No SMOTE)",
            SVC(
                C=0.8,
                gamma="scale",
                probability=True,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            "Gradient Boosting (No SMOTE)",
            GradientBoostingClassifier(
                n_estimators=240,
                learning_rate=0.025,
                max_depth=2,
                min_samples_leaf=15,
                subsample=0.85,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "HistGradientBoosting (No SMOTE Tuned)",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.018,
                max_leaf_nodes=8,
                min_samples_leaf=20,
                l2_regularization=0.40,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "XGBoost (SMOTE)",
            XGBClassifier(
                n_estimators=220,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.90,
                colsample_bytree=0.90,
                eval_metric="logloss",
                n_jobs=1,
                random_state=random_state,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "XGBoost (No SMOTE Tuned)",
            XGBClassifier(
                n_estimators=520,
                max_depth=2,
                learning_rate=0.018,
                subsample=0.85,
                colsample_bytree=0.75,
                min_child_weight=12,
                reg_lambda=10,
                reg_alpha=0.3,
                eval_metric="logloss",
                n_jobs=1,
                random_state=random_state,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "LightGBM (SMOTE Balanced)",
            LGBMClassifier(
                n_estimators=220,
                learning_rate=0.05,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "LightGBM (No SMOTE Tuned)",
            LGBMClassifier(
                n_estimators=650,
                learning_rate=0.015,
                num_leaves=7,
                min_child_samples=70,
                subsample=0.85,
                colsample_bytree=0.70,
                reg_lambda=12,
                reg_alpha=0.2,
                random_state=random_state,
                verbose=-1,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "LightGBM (No SMOTE Balanced)",
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.025,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.90,
                colsample_bytree=0.80,
                reg_lambda=4,
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
            scale_numeric=False,
        ),
        ModelSpec(
            "CatBoost (SMOTE)",
            CatBoostClassifier(
                iterations=220,
                depth=4,
                learning_rate=0.05,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            ),
            sampler="smote",
        ),
        ModelSpec(
            "CatBoost (No SMOTE Tuned)",
            CatBoostClassifier(
                iterations=450,
                depth=3,
                learning_rate=0.025,
                l2_leaf_reg=8,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            ),
        ),
        ModelSpec(
            "CatBoost (No SMOTE Balanced)",
            CatBoostClassifier(
                iterations=450,
                depth=3,
                learning_rate=0.025,
                l2_leaf_reg=8,
                auto_class_weights="Balanced",
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            ),
        ),
    ]


def build_model_pipeline(
    estimator: object,
    x_train,
    random_state: int,
    sampler: Literal["smote"] | None = "smote",
    encoder: Literal["onehot", "target"] = "onehot",
    scale_numeric: bool = True,
    drop_columns: tuple[str, ...] = (),
) -> ImbPipeline | Pipeline:
    """Create a leakage-safe preprocessing + optional resampling + model pipeline."""

    steps = [
        (
            "preprocessor",
            build_preprocessor(
                x_train,
                encoder=encoder,
                scale_numeric=scale_numeric,
                random_state=random_state,
                drop_columns=drop_columns,
            ),
        )
    ]
    if sampler == "smote":
        steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("model", clone(estimator)))
        return ImbPipeline(steps=steps)
    steps.append(("model", clone(estimator)))
    return Pipeline(steps=steps)


def build_soft_voting_ensemble(
    top_specs: list[ModelSpec],
    x_train,
    random_state: int,
) -> VotingClassifier:
    """Build a soft-voting ensemble from the top individual model specifications."""

    estimators = []
    for spec in top_specs:
        name = (
            spec.name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        estimators.append(
            (
                name,
                build_model_pipeline(
                    spec.estimator,
                    x_train,
                    random_state,
                    sampler=spec.sampler,
                    encoder=spec.encoder,
                    scale_numeric=spec.scale_numeric,
                    drop_columns=spec.drop_columns,
                ),
            )
        )

    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)


def build_stacking_ensemble(
    top_specs: list[ModelSpec],
    x_train,
    random_state: int,
) -> StackingClassifier:
    """Build a logistic meta-model over the top individual model pipelines."""

    estimators = []
    for spec in top_specs:
        name = (
            spec.name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        estimators.append(
            (
                name,
                build_model_pipeline(
                    spec.estimator,
                    x_train,
                    random_state,
                    sampler=spec.sampler,
                    encoder=spec.encoder,
                    scale_numeric=spec.scale_numeric,
                    drop_columns=spec.drop_columns,
                ),
            )
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, C=0.5),
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,
    )


def build_focused_voting_ensemble(x_train, random_state: int) -> VotingClassifier:
    """Build the accuracy-focused ensemble found during iterative experimentation."""

    estimators = [
        (
            "lr_focused",
            build_model_pipeline(
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
                x_train,
                random_state,
                sampler=None,
                encoder="onehot",
                scale_numeric=True,
                drop_columns=FOCUSED_ACCURACY_DROP_COLUMNS,
            ),
        ),
        (
            "lr_drop_sparse_target",
            build_model_pipeline(
                LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
                x_train,
                random_state,
                sampler=None,
                encoder="target",
                scale_numeric=True,
                drop_columns=DROP_SPARSE_INTERACTION_COLUMNS,
            ),
        ),
        (
            "lgbm_all",
            build_model_pipeline(
                LGBMClassifier(
                    n_estimators=650,
                    learning_rate=0.015,
                    num_leaves=7,
                    min_child_samples=70,
                    subsample=0.85,
                    colsample_bytree=0.70,
                    reg_lambda=12,
                    reg_alpha=0.2,
                    random_state=random_state,
                    verbose=-1,
                ),
                x_train,
                random_state,
                sampler=None,
                encoder="onehot",
                scale_numeric=False,
            ),
        ),
        (
            "hgb_no_demographics",
            build_model_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.018,
                    max_leaf_nodes=8,
                    min_samples_leaf=20,
                    l2_regularization=0.40,
                    random_state=random_state,
                ),
                x_train,
                random_state,
                sampler=None,
                encoder="onehot",
                scale_numeric=False,
                drop_columns=NO_DEMOGRAPHIC_DROP_COLUMNS,
            ),
        ),
        (
            "xgb_all",
            build_model_pipeline(
                XGBClassifier(
                    n_estimators=520,
                    max_depth=2,
                    learning_rate=0.018,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    min_child_weight=12,
                    reg_lambda=10,
                    reg_alpha=0.3,
                    eval_metric="logloss",
                    n_jobs=1,
                    random_state=random_state,
                ),
                x_train,
                random_state,
                sampler=None,
                encoder="onehot",
                scale_numeric=False,
            ),
        ),
    ]
    return VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[4, 1, 1, 1, 1],
        n_jobs=1,
    )
