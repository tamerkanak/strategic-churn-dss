from __future__ import annotations

from src.churn_dss.recommendations import recommendation_for_customer, risk_tier


def test_risk_tier_boundaries() -> None:
    assert risk_tier(0.29) == "low"
    assert risk_tier(0.30) == "medium"
    assert risk_tier(0.59) == "medium"
    assert risk_tier(0.60) == "high"


def test_recommendation_prioritizes_contract_migration() -> None:
    rec = recommendation_for_customer(
        {
            "Contract": "Month-to-month",
            "MonthlyCharges": 95,
            "tenure": 4,
            "PaymentMethod": "Electronic check",
            "OnlineSecurity": "No",
            "TechSupport": "No",
        }
    )

    assert rec.action == "Contract migration offer"


def test_recommendation_uses_support_when_contract_is_stable() -> None:
    rec = recommendation_for_customer(
        {
            "Contract": "Two year",
            "MonthlyCharges": 55,
            "tenure": 36,
            "PaymentMethod": "Mailed check",
            "OnlineSecurity": "No",
            "TechSupport": "Yes",
        }
    )

    assert rec.action == "Technical support outreach"
