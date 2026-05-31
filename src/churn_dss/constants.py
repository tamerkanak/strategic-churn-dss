"""Shared constants for the churn DSS project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = Path(os.getenv("CHURN_DSS_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
REPORT_DIR = Path(os.getenv("CHURN_DSS_REPORT_DIR", PROJECT_ROOT / "reports"))
FIGURE_DIR = REPORT_DIR / "figures"

DEFAULT_DATA_URL = os.getenv(
    "CHURN_DSS_DATA_URL",
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/"
    "Telco-Customer-Churn.csv",
)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

LOW_RISK_CUTOFF = 0.30
HIGH_RISK_CUTOFF = 0.60
FALSE_NEGATIVE_COST = 5
FALSE_POSITIVE_COST = 1

REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

LOCAL_DATA_CANDIDATES = (
    "Telco-Customer-Churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "telco_customer_churn.csv",
    "Telco_customer_churn.csv",
)
