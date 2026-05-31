"""Feature lists and preprocessing pipeline construction."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from .constants import ID_COLUMN


def feature_columns(x: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding identifiers."""

    return [column for column in x.columns if column != ID_COLUMN]


def split_feature_types(
    x: pd.DataFrame,
    drop_columns: tuple[str, ...] = (),
) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical model inputs."""

    dropped = set(drop_columns)
    model_columns = [column for column in feature_columns(x) if column not in dropped]
    numeric_columns = [
        column for column in model_columns if pd.api.types.is_numeric_dtype(x[column])
    ]
    categorical_columns = [column for column in model_columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(
    x: pd.DataFrame,
    encoder: Literal["onehot", "target"] = "onehot",
    scale_numeric: bool = True,
    random_state: int = 42,
    drop_columns: tuple[str, ...] = (),
) -> ColumnTransformer:
    """Build a leakage-safe preprocessing transformer."""

    numeric_columns, categorical_columns = split_feature_types(x, drop_columns=drop_columns)

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(
        steps=numeric_steps,
    )

    if encoder == "target":
        categorical_encoder = TargetEncoder(
            target_type="binary",
            smooth=20.0,
            random_state=random_state,
        )
    else:
        categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", categorical_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
