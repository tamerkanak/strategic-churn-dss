from __future__ import annotations

import numpy as np
import pandas as pd

from src.churn_dss.data import add_domain_features, clean_dataset


def raw_telco_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 25],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "Yes"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Credit card (automatic)"],
            "MonthlyCharges": [29.85, 89.10],
            "TotalCharges": [" ", "2100.5"],
            "Churn": ["No", "Yes"],
        }
    )


def test_clean_dataset_converts_target_and_total_charges() -> None:
    clean = clean_dataset(raw_telco_sample())

    assert clean["Churn"].tolist() == [0, 1]
    assert np.isnan(clean.loc[0, "TotalCharges"])
    assert clean.loc[1, "TotalCharges"] == 2100.5


def test_add_domain_features_creates_interpretable_columns() -> None:
    featured = add_domain_features(clean_dataset(raw_telco_sample()))

    assert featured.loc[0, "is_month_to_month"] == 1
    assert featured.loc[1, "has_auto_payment"] == 1
    assert featured.loc[1, "support_security_bundle"] == 1
    assert featured.loc[0, "tenure_band"] == "0-12 months"
    assert featured.loc[1, "total_active_services"] == 6
