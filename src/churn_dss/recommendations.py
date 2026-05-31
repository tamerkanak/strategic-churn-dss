"""Risk tiering and retention recommendation rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .constants import HIGH_RISK_CUTOFF, LOW_RISK_CUTOFF


@dataclass(frozen=True)
class Recommendation:
    action: str
    rationale: str


def risk_tier(probability: float) -> str:
    """Map churn probability to an operational risk tier."""

    if probability < LOW_RISK_CUTOFF:
        return "low"
    if probability < HIGH_RISK_CUTOFF:
        return "medium"
    return "high"


def recommendation_for_customer(row: Mapping[str, object] | pd.Series) -> Recommendation:
    """Return the highest-priority managerial action for a customer profile."""

    value = row.get

    contract = value("Contract", "")
    monthly_charges = float(value("MonthlyCharges", 0) or 0)
    tenure = float(value("tenure", 0) or 0)
    payment_method = str(value("PaymentMethod", ""))
    online_security = value("OnlineSecurity", "")
    tech_support = value("TechSupport", "")

    if contract == "Month-to-month":
        return Recommendation(
            action="Contract migration offer",
            rationale="Month-to-month customers have weaker switching barriers; offer a one-year plan incentive.",
        )
    if monthly_charges >= 80:
        return Recommendation(
            action="Pricing and bundle review",
            rationale="High monthly charges may create price sensitivity; review plan fit or targeted discount.",
        )
    if online_security != "Yes" or tech_support != "Yes":
        return Recommendation(
            action="Technical support outreach",
            rationale="Missing support/security services suggest a service-quality intervention opportunity.",
        )
    if "Electronic check" in payment_method:
        return Recommendation(
            action="Auto-payment incentive",
            rationale="Electronic-check customers often show elevated churn risk; encourage automated payment.",
        )
    if tenure <= 12:
        return Recommendation(
            action="Onboarding outreach",
            rationale="Early-lifecycle customers need onboarding reinforcement before dissatisfaction accumulates.",
        )
    return Recommendation(
        action="Loyalty reinforcement",
        rationale="Maintain the customer relationship through loyalty benefits and service-quality monitoring.",
    )


def add_decision_outputs(
    customers: pd.DataFrame,
    probabilities: list[float] | pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Add probability, risk tier, intervention flag, and recommendation columns."""

    result = customers.copy()
    result["churn_probability"] = list(probabilities)
    result["risk_tier"] = result["churn_probability"].map(risk_tier)
    result["intervention_candidate"] = result["churn_probability"] >= threshold
    recommendations = result.apply(recommendation_for_customer, axis=1)
    result["recommended_action"] = [item.action for item in recommendations]
    result["recommendation_rationale"] = [item.rationale for item in recommendations]
    return result
