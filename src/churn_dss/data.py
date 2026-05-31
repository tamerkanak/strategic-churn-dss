"""Data loading, cleaning, and feature engineering entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_DATA_URL,
    ID_COLUMN,
    LOCAL_DATA_CANDIDATES,
    RAW_DATA_DIR,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
)

DataSource = Literal["auto", "local", "url"]


def _validate_dataset_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def find_local_dataset(raw_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Return the first local CSV that matches the expected Telco schema."""

    if not raw_dir.exists():
        return None

    candidates = [raw_dir / name for name in LOCAL_DATA_CANDIDATES]
    candidates.extend(sorted(raw_dir.glob("*.csv")))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        try:
            sample = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        if REQUIRED_COLUMNS.issubset(sample.columns):
            return candidate
    return None


def load_dataset(data_source: str = "auto", data_url: str = DEFAULT_DATA_URL) -> tuple[pd.DataFrame, str]:
    """Load the Telco churn dataset from a local CSV, URL, or auto-discovery."""

    raw_dir = RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(data_source)
    if data_source not in {"auto", "local", "url"} and source_path.exists():
        df = pd.read_csv(source_path)
        _validate_dataset_columns(df)
        return df, str(source_path)

    if data_source in {"auto", "local"}:
        local_file = find_local_dataset(raw_dir)
        if local_file is not None:
            df = pd.read_csv(local_file)
            _validate_dataset_columns(df)
            return df, str(local_file)
        if data_source == "local":
            raise FileNotFoundError(f"No compatible Telco CSV found in {raw_dir}")

    df = pd.read_csv(data_url)
    _validate_dataset_columns(df)
    cached_path = raw_dir / "Telco-Customer-Churn.csv"
    if not cached_path.exists():
        df.to_csv(cached_path, index=False)
    return df, data_url


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw IBM Telco data while preserving customer identifiers."""

    clean = df.copy()
    clean.columns = [str(column).strip() for column in clean.columns]

    for column in clean.select_dtypes(include=["object"]).columns:
        clean[column] = clean[column].astype(str).str.strip()
        clean[column] = clean[column].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    clean["MonthlyCharges"] = pd.to_numeric(clean["MonthlyCharges"], errors="coerce")
    clean["tenure"] = pd.to_numeric(clean["tenure"], errors="coerce")
    clean["SeniorCitizen"] = pd.to_numeric(clean["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    if clean.duplicated(subset=[ID_COLUMN]).any():
        clean = clean.drop_duplicates(subset=[ID_COLUMN], keep="first")

    clean = clean.dropna(subset=[TARGET_COLUMN])
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)
    return clean


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable churn features used by both training and dashboard scoring."""

    featured = df.copy()
    featured["TotalCharges"] = pd.to_numeric(featured["TotalCharges"], errors="coerce")
    featured["MonthlyCharges"] = pd.to_numeric(featured["MonthlyCharges"], errors="coerce")
    featured["tenure"] = pd.to_numeric(featured["tenure"], errors="coerce")

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    featured["total_active_services"] = (
        featured[service_columns].apply(lambda row: sum(value == "Yes" for value in row), axis=1).astype(int)
    )
    featured["is_month_to_month"] = (featured["Contract"] == "Month-to-month").astype(int)
    featured["has_auto_payment"] = (
        featured["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    )
    featured["support_security_bundle"] = (
        (featured["OnlineSecurity"] == "Yes") & (featured["TechSupport"] == "Yes")
    ).astype(int)
    featured["avg_total_charge_per_month"] = featured["TotalCharges"] / featured["tenure"].clip(lower=1)
    featured["charge_gap_vs_average"] = (
        featured["MonthlyCharges"] - featured["avg_total_charge_per_month"]
    )
    featured["monthly_charge_log"] = np.log1p(featured["MonthlyCharges"])
    featured["total_charge_log"] = np.log1p(
        featured["TotalCharges"].fillna(featured["TotalCharges"].median())
    )
    featured["monthly_x_tenure"] = featured["MonthlyCharges"] * featured["tenure"]
    featured["charge_per_service"] = featured["MonthlyCharges"] / featured[
        "total_active_services"
    ].replace(0, 1)
    featured["tenure_inverse"] = 1 / (featured["tenure"] + 1)
    featured["has_internet"] = (featured["InternetService"] != "No").astype(int)
    featured["has_fiber"] = (featured["InternetService"] == "Fiber optic").astype(int)
    featured["fiber_no_security"] = (
        (featured["InternetService"] == "Fiber optic") & (featured["OnlineSecurity"] != "Yes")
    ).astype(int)
    featured["fiber_no_support"] = (
        (featured["InternetService"] == "Fiber optic") & (featured["TechSupport"] != "Yes")
    ).astype(int)
    featured["fiber_no_security_support"] = (
        (featured["InternetService"] == "Fiber optic")
        & (featured["OnlineSecurity"] != "Yes")
        & (featured["TechSupport"] != "Yes")
    ).astype(int)
    featured["month_to_month_fiber"] = (
        (featured["Contract"] == "Month-to-month") & (featured["InternetService"] == "Fiber optic")
    ).astype(int)
    featured["electronic_check_m2m"] = (
        (featured["PaymentMethod"] == "Electronic check")
        & (featured["Contract"] == "Month-to-month")
    ).astype(int)
    featured["short_tenure_high_charge"] = (
        (featured["tenure"] <= 12) & (featured["MonthlyCharges"] >= 70)
    ).astype(int)
    featured["new_customer"] = (featured["tenure"] <= 6).astype(int)
    featured["very_low_tenure"] = (featured["tenure"] <= 2).astype(int)
    featured["very_long_tenure"] = (featured["tenure"] >= 60).astype(int)
    featured["streaming_count"] = (
        (featured["StreamingTV"] == "Yes").astype(int)
        + (featured["StreamingMovies"] == "Yes").astype(int)
    )
    featured["backup_protection_count"] = (
        (featured["OnlineBackup"] == "Yes").astype(int)
        + (featured["DeviceProtection"] == "Yes").astype(int)
    )
    featured["support_count"] = (
        (featured["OnlineSecurity"] == "Yes").astype(int)
        + (featured["TechSupport"] == "Yes").astype(int)
    )
    featured["risk_service_gap_count"] = (
        (featured["OnlineSecurity"] != "Yes").astype(int)
        + (featured["TechSupport"] != "Yes").astype(int)
        + (featured["OnlineBackup"] != "Yes").astype(int)
        + (featured["DeviceProtection"] != "Yes").astype(int)
    )
    featured["contract_commitment_score"] = (
        featured["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0)
    )
    featured["payment_risk_score"] = (
        featured["PaymentMethod"]
        .map(
            {
                "Electronic check": 3,
                "Mailed check": 2,
                "Bank transfer (automatic)": 1,
                "Credit card (automatic)": 1,
            }
        )
        .fillna(2)
    )
    featured["contract_x_tenure"] = featured["contract_commitment_score"] * featured["tenure"]
    featured["contract_x_monthly"] = (
        featured["contract_commitment_score"] * featured["MonthlyCharges"]
    )
    featured["support_x_fiber"] = featured["support_count"] * featured["has_fiber"]
    featured["services_x_tenure"] = featured["total_active_services"] * featured["tenure"]
    featured["is_senior_m2m"] = (
        (featured["SeniorCitizen"] == 1) & (featured["Contract"] == "Month-to-month")
    ).astype(int)
    featured["dependents_partner_score"] = (
        (featured["Partner"] == "Yes").astype(int) + (featured["Dependents"] == "Yes").astype(int)
    )
    featured["paperless_echeck"] = (
        (featured["PaperlessBilling"] == "Yes") & (featured["PaymentMethod"] == "Electronic check")
    ).astype(int)
    featured["no_protection_services"] = (
        (featured["OnlineSecurity"] != "Yes")
        & (featured["OnlineBackup"] != "Yes")
        & (featured["DeviceProtection"] != "Yes")
        & (featured["TechSupport"] != "Yes")
    ).astype(int)
    featured["internet_service_group"] = (
        featured["InternetService"].astype(str) + "_" + featured["Contract"].astype(str)
    )
    featured["payment_contract_group"] = (
        featured["PaymentMethod"].astype(str) + "_" + featured["Contract"].astype(str)
    )
    featured["tenure_band"] = pd.cut(
        featured["tenure"],
        bins=[-1, 12, 24, 48, 72, np.inf],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months", "72+ months"],
    ).astype("object")
    featured["tenure_bucket"] = pd.cut(
        featured["tenure"],
        bins=[-1, 3, 6, 12, 24, 48, 72, np.inf],
        labels=["0-3", "4-6", "7-12", "13-24", "25-48", "49-72", "72+"],
    ).astype("object")
    featured["monthly_charge_band"] = pd.cut(
        featured["MonthlyCharges"],
        bins=[-np.inf, 35, 60, 85, np.inf],
        labels=["low", "moderate", "high", "premium"],
    ).astype("object")
    featured["tenure_charge_band"] = (
        featured["tenure_bucket"].astype(str) + "_" + featured["monthly_charge_band"].astype(str)
    )
    return featured


def prepare_dataset(data_source: str = "auto") -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    """Load, clean, enrich, and split into model features/target/customer ids."""

    raw_df, source = load_dataset(data_source=data_source)
    clean = clean_dataset(raw_df)
    featured = add_domain_features(clean)
    y = featured[TARGET_COLUMN].astype(int)
    customer_ids = featured[ID_COLUMN].copy()
    x = featured.drop(columns=[TARGET_COLUMN])
    return x, y, customer_ids, source


def tenure_summary(featured_df: pd.DataFrame) -> pd.DataFrame:
    """Create lifecycle-oriented churn summary by tenure band."""

    if TARGET_COLUMN not in featured_df.columns:
        raise ValueError(f"{TARGET_COLUMN} must be present for tenure summary.")

    grouped = (
        featured_df.groupby("tenure_band", observed=False)
        .agg(customers=(TARGET_COLUMN, "size"), churn_rate=(TARGET_COLUMN, "mean"))
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churn_rate"].fillna(0)
    return grouped
