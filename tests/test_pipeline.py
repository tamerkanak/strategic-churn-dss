from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.churn_dss.data import add_domain_features, clean_dataset
from src.churn_dss.models import build_model_pipeline


def _synthetic_telco(rows: int = 18) -> pd.DataFrame:
    records = []
    for index in range(rows):
        churn = index % 2 == 0
        records.append(
            {
                "customerID": f"C{index:03d}",
                "gender": "Female" if index % 3 == 0 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 4 == 0 else "No",
                "Dependents": "No",
                "tenure": index + 1,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if churn else "No",
                "InternetService": "Fiber optic" if churn else "DSL",
                "OnlineSecurity": "No" if churn else "Yes",
                "OnlineBackup": "No" if churn else "Yes",
                "DeviceProtection": "No" if churn else "Yes",
                "TechSupport": "No" if churn else "Yes",
                "StreamingTV": "Yes" if churn else "No",
                "StreamingMovies": "Yes" if churn else "No",
                "Contract": "Month-to-month" if churn else "Two year",
                "PaperlessBilling": "Yes" if churn else "No",
                "PaymentMethod": "Electronic check" if churn else "Credit card (automatic)",
                "MonthlyCharges": 90.0 if churn else 45.0,
                "TotalCharges": str((index + 1) * 50.0),
                "Churn": "Yes" if churn else "No",
            }
        )
    return pd.DataFrame(records)


def test_pipeline_fits_and_predicts_probabilities() -> None:
    clean = clean_dataset(_synthetic_telco())
    featured = add_domain_features(clean)
    y = featured["Churn"]
    x = featured.drop(columns=["Churn"])

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=500, solver="liblinear"),
        x,
        random_state=42,
    )
    pipeline.fit(x, y)
    probabilities = pipeline.predict_proba(x)[:, 1]

    assert probabilities.shape == (len(x),)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
