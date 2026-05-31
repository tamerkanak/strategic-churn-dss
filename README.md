# Strategic DSS for Customer Churn Prediction

This project completes the Phase 2 implementation for **A Strategic Decision Support System for Customer Churn Prediction**. It trains and evaluates churn prediction models on the IBM Telco Customer Churn dataset, explains churn drivers, maps customers into operational risk tiers, and presents the results in a Streamlit decision-support dashboard.

## Project Outputs

- Leakage-safe machine learning pipeline with preprocessing, SMOTE, model comparison, threshold analysis, SHAP/LIME explanations, and retention recommendations.
- Streamlit dashboard for model performance, lifecycle analysis, customer-level prediction, churn drivers, and suggested managerial actions.
- Phase 2 final report in editable Markdown, DOCX, and PDF formats.
- Automated tests for data preparation, risk tiers, recommendations, and pipeline smoke behavior.

## Setup

```powershell
py -m pip install -r requirements.txt
```

The code reads the dataset from `data/raw/` first. If no compatible local CSV is found, it downloads the public IBM Telco CSV automatically from:

```text
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Usage

Train models, evaluate them, and create artifacts:

```powershell
py -m src.churn_dss.train --data-source auto --random-state 42
```

Generate the final report after training:

```powershell
py -m src.churn_dss.generate_report
```

Run the decision-support dashboard:

```powershell
py -m streamlit run app/streamlit_app.py
```

Run tests and lint checks:

```powershell
py -m pytest
py -m ruff check .
```

## Environment Variables

No environment variables are required.

Optional:

- `CHURN_DSS_DATA_URL`: overrides the default IBM CSV URL.
- `CHURN_DSS_ARTIFACT_DIR`: overrides the default `artifacts/` output directory.
- `CHURN_DSS_REPORT_DIR`: overrides the default `reports/` output directory.

## Main Files

- `src/churn_dss/train.py`: end-to-end training, evaluation, thresholding, explanation, and artifact generation.
- `src/churn_dss/data.py`: dataset discovery, download, validation, cleaning, and feature engineering entry point.
- `src/churn_dss/features.py`: model feature construction and preprocessing definitions.
- `src/churn_dss/recommendations.py`: risk tiers and retention action rules.
- `src/churn_dss/explainability.py`: SHAP/LIME explanation artifacts.
- `src/churn_dss/generate_report.py`: Phase 2 report generation.
- `app/streamlit_app.py`: interactive DSS dashboard.

## Contributing

Keep the implementation reproducible and course-deliverable oriented:

1. Put reusable logic in `src/churn_dss/` instead of embedding it in the dashboard.
2. Add or update tests when changing preprocessing, model outputs, thresholds, or recommendation rules.
3. Keep generated artifacts in `artifacts/` and final report files in `reports/`.
4. Do not base operational recommendations on demographic variables alone.
