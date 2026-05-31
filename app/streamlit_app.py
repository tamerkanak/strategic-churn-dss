"""Streamlit dashboard for the strategic churn decision support system."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

PAGE_OPTIONS = ("Overview", "Model Performance", "Customer Desk", "Drivers & Actions")
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def _json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_artifacts() -> dict[str, object]:
    required = {
        "model_comparison": ARTIFACT_DIR / "model_comparison.csv",
        "thresholds": ARTIFACT_DIR / "threshold_analysis.csv",
        "decisions": ARTIFACT_DIR / "decision_outputs.csv",
        "tenure": ARTIFACT_DIR / "tenure_summary.csv",
        "shap": ARTIFACT_DIR / "shap_global_importance.csv",
        "shap_local": ARTIFACT_DIR / "shap_local_explanations.csv",
        "lime_local": ARTIFACT_DIR / "lime_local_explanations.csv",
        "final_metrics": ARTIFACT_DIR / "final_metrics.json",
        "dataset_summary": ARTIFACT_DIR / "dataset_summary.json",
    }
    missing = [path for path in required.values() if not path.exists()]
    if missing:
        return {"missing": missing}

    return {
        "model_comparison": pd.read_csv(required["model_comparison"]),
        "thresholds": pd.read_csv(required["thresholds"]),
        "decisions": pd.read_csv(required["decisions"]),
        "tenure": pd.read_csv(required["tenure"]),
        "shap": pd.read_csv(required["shap"]),
        "shap_local": pd.read_csv(required["shap_local"]),
        "lime_local": pd.read_csv(required["lime_local"]),
        "final_metrics": _json(required["final_metrics"]),
        "dataset_summary": _json(required["dataset_summary"]),
    }


def risk_color(tier: str) -> str:
    return {"low": "#178a52", "medium": "#b7791f", "high": "#c2412d"}.get(tier, "#42526e")


def configure_page() -> None:
    st.set_page_config(
        page_title="Strategic Churn DSS",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #f4f7fb;
            --surface: #ffffff;
            --surface-strong: #eef3f8;
            --ink: #102033;
            --muted: #66758a;
            --border: #dbe5ef;
            --blue: #1f6feb;
            --teal: #0f766e;
            --green: #178a52;
            --amber: #b7791f;
            --red: #c2412d;
            --shadow: 0 18px 50px rgba(16, 32, 51, 0.08);
            --radius: 8px;
        }

        #MainMenu,
        footer,
        header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="collapsedControl"],
        .stDeployButton,
        section[data-testid="stSidebar"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .stApp {
            background:
                linear-gradient(180deg, rgba(235, 242, 250, 0.95) 0%, rgba(244, 247, 251, 1) 42%),
                #f4f7fb !important;
            color: var(--ink);
        }

        .block-container {
            max-width: 1220px;
            padding: 1.1rem 1.6rem 2.8rem;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            letter-spacing: 0;
        }

        h1, h2, h3, h4, p, label, span {
            color: var(--ink);
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: #ffffff;
            background: linear-gradient(135deg, #102033 0%, #1f6feb 100%);
            box-shadow: var(--shadow);
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        .brand-title {
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .brand-subtitle,
        .microcopy {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .status-strip {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 0.55rem;
        }

        .status-token {
            border: 1px solid var(--border);
            border-radius: 999px;
            background: rgba(255,255,255,0.82);
            color: #34455c;
            padding: 0.45rem 0.68rem;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .hero-panel {
            border: 1px solid rgba(31, 111, 235, 0.14);
            border-radius: var(--radius);
            background:
                linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(244,248,252,0.98) 55%, rgba(232,241,255,0.92) 100%);
            box-shadow: var(--shadow);
            padding: 1.35rem 1.45rem;
            margin-bottom: 1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
            gap: 1rem;
            align-items: center;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(1.9rem, 4vw, 3.15rem);
            line-height: 1.02;
            font-weight: 850;
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero-text {
            margin: 0.7rem 0 0;
            max-width: 760px;
            color: #516174;
            font-size: 0.98rem;
            line-height: 1.6;
        }

        .hero-side {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: #ffffff;
            padding: 1rem;
        }

        .hero-side-title {
            font-size: 0.78rem;
            color: var(--muted);
            text-transform: uppercase;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .hero-side-value {
            font-size: 1.8rem;
            font-weight: 850;
            color: var(--ink);
            line-height: 1.1;
        }

        .hero-side-detail {
            margin-top: 0.4rem;
            color: var(--muted);
            font-size: 0.82rem;
        }

        div[data-testid="stSegmentedControl"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.35rem;
            margin: 0.55rem 0 1.1rem;
            box-shadow: 0 10px 28px rgba(16, 32, 51, 0.05);
        }

        div[data-testid="stSegmentedControl"] label {
            border-radius: 7px !important;
            font-weight: 750 !important;
            color: #334155 !important;
        }

        div[data-baseweb="button-group"] {
            display: inline-flex;
            width: auto;
            max-width: 100%;
            overflow: hidden;
            background: #ffffff !important;
            border: 1px solid var(--border);
            border-radius: 9px;
            padding: 0.28rem;
            gap: 0.18rem;
            box-shadow: 0 12px 30px rgba(16, 32, 51, 0.06);
        }

        div[data-baseweb="button-group"] button[kind^="segmented_control"] {
            min-height: 38px;
            border: 0 !important;
            border-radius: 7px !important;
            background: transparent !important;
            color: #526174 !important;
            padding: 0.42rem 0.95rem !important;
            box-shadow: none !important;
            transition: background-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
        }

        div[data-baseweb="button-group"] button[kind^="segmented_control"] p {
            color: inherit !important;
            font-size: 0.86rem;
            font-weight: 760;
            line-height: 1.2;
        }

        div[data-baseweb="button-group"] button[kind="segmented_controlActive"] {
            background: #eef6ff !important;
            color: #102033 !important;
            box-shadow: inset 0 0 0 1px #bcd7ff !important;
        }

        div[data-baseweb="button-group"] button[kind="segmented_control"]:hover {
            background: #f4f7fb !important;
            color: #102033 !important;
        }

        .section-heading {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            gap: 1rem;
            margin: 1.1rem 0 0.55rem;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 850;
            color: var(--ink);
            margin: 0;
        }

        .section-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin: 0.1rem 0 0;
        }

        .metric-card,
        .content-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: 0 14px 36px rgba(16, 32, 51, 0.06);
        }

        .metric-card {
            padding: 0.95rem 1rem;
            min-height: 108px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.76rem;
            line-height: 1.25;
            font-weight: 800;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--ink);
            margin-top: 0.42rem;
            font-size: clamp(1.3rem, 2.5vw, 1.9rem);
            font-weight: 850;
            line-height: 1.08;
        }

        .metric-detail {
            color: var(--muted);
            margin-top: 0.38rem;
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .risk-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.32rem 0.58rem;
            border-radius: 6px;
            color: white;
            font-size: 0.75rem;
            font-weight: 850;
            text-transform: uppercase;
        }

        .profile-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin-top: 0.35rem;
        }

        .profile-item {
            border: 1px solid var(--border);
            border-radius: 7px;
            background: #ffffff;
            padding: 0.72rem 0.78rem;
        }

        .profile-label {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 800;
        }

        .profile-value {
            color: var(--ink);
            margin-top: 0.25rem;
            font-size: 0.92rem;
            font-weight: 750;
            word-break: break-word;
        }

        .driver-list {
            display: grid;
            gap: 0.45rem;
            margin-top: 0.35rem;
        }

        .driver-item {
            border: 1px solid var(--border);
            border-radius: 7px;
            background: #ffffff;
            padding: 0.65rem 0.75rem;
            color: #334155;
            font-size: 0.86rem;
            line-height: 1.38;
        }

        .recommendation-box {
            border: 1px solid #cfe2ff;
            border-radius: var(--radius);
            background: #eef6ff;
            color: #16324f;
            padding: 0.95rem 1rem;
            line-height: 1.55;
        }

        .stPlotlyChart,
        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: #ffffff;
            box-shadow: 0 14px 36px rgba(16, 32, 51, 0.045);
            overflow: hidden;
        }

        [data-baseweb="select"] > div,
        [data-testid="stCheckbox"] label,
        .stSlider {
            color: var(--ink) !important;
        }

        [data-baseweb="select"] > div {
            background: #ffffff !important;
            border-color: var(--border) !important;
            border-radius: 8px !important;
        }

        div[data-testid="stAlert"] {
            border-radius: var(--radius);
            border: 1px solid #cfe2ff;
            background: #eef6ff;
            color: #16324f;
        }

        .stMarkdown a {
            color: var(--blue);
        }

        @media (max-width: 900px) {
            .block-container { padding: 0.75rem 0.9rem 2rem; }
            .topbar, .section-heading { align-items: flex-start; flex-direction: column; }
            .status-strip { justify-content: flex-start; }
            .hero-grid { grid-template-columns: 1fr; }
            .profile-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_missing(missing: list[Path]) -> None:
    st.error("Training artifacts are missing. Run the training command before opening the dashboard.")
    st.code("py -m src.churn_dss.train --data-source auto --random-state 42", language="powershell")
    with st.expander("Missing files"):
        for path in missing:
            st.write(str(path))


def fmt_pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_shell_header(artifacts: dict[str, object]) -> None:
    final_metrics = artifacts["final_metrics"]
    dataset_summary = artifacts["dataset_summary"]
    selected = final_metrics["selected_model"]
    threshold = final_metrics["threshold_summary"]["cost_sensitive_threshold"]
    default = final_metrics["default_threshold_metrics"]

    st.markdown(
        dedent(
            f"""
        <div class="topbar">
            <div class="brand-lockup">
                <div class="brand-mark">CD</div>
                <div>
                    <div class="brand-title">Churn Decision Studio</div>
                    <div class="brand-subtitle">Strategic Information Systems / Phase 2</div>
                </div>
            </div>
            <div class="status-strip">
                <span class="status-token">IBM Telco</span>
                <span class="status-token">{dataset_summary["test_rows"]:,} hold-out cases</span>
                <span class="status-token">Threshold {threshold:.2f}</span>
            </div>
        </div>
        <section class="hero-panel">
            <div class="hero-grid">
                <div>
                    <h1 class="hero-title">Strategic churn command center</h1>
                    <p class="hero-text">
                        Retention risk, model performance, customer explanations, and action planning
                        are organized as one analyst-ready decision surface.
                    </p>
                </div>
                <div class="hero-side">
                    <div class="hero-side-title">Operational model</div>
                    <div class="hero-side-value">{escape(selected)}</div>
                    <div class="hero-side-detail">
                        Accuracy {default["accuracy"]:.3f} / PR-AUC {default["pr_auc"]:.3f}
                    </div>
                </div>
            </div>
        </section>
        """
        ).strip(),
        unsafe_allow_html=True,
    )


def render_navigation() -> str:
    default_page = st.session_state.get("selected_page", PAGE_OPTIONS[0])
    selected_page = st.segmented_control(
        "View",
        PAGE_OPTIONS,
        default=default_page,
        label_visibility="collapsed",
        key="page_selector",
    )
    if selected_page is None:
        selected_page = default_page
    st.session_state["selected_page"] = selected_page
    return selected_page


def section_heading(title: str, note: str | None = None) -> None:
    note_html = f'<p class="section-note">{escape(note)}</p>' if note else ""
    st.markdown(
        (
            f'<div class="section-heading"><div><h2 class="section-title">{escape(title)}</h2>'
            f"{note_html}</div></div>"
        ),
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str | None = None) -> str:
    detail_html = f'<div class="metric-detail">{escape(detail)}</div>' if detail else ""
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f"{detail_html}"
        "</div>"
    )


def render_metric_cards(cards: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(cards))
    for column, (label, value, detail) in zip(columns, cards, strict=True):
        column.markdown(metric_card(label, value, detail), unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#1f2937", "size": 13},
        margin={"l": 22, "r": 18, "t": 42, "b": 28},
        title={"font": {"size": 16, "color": "#0f172a"}, "x": 0.02, "xanchor": "left"},
        legend={
            "orientation": "h",
            "y": -0.18,
            "x": 0,
            "font": {"color": "#1f2937", "size": 12},
        },
    )
    axis_style = {
        "gridcolor": "#e5edf5",
        "zerolinecolor": "#cbd5e1",
        "linecolor": "#94a3b8",
        "tickcolor": "#64748b",
        "tickfont": {"color": "#1f2937", "size": 12},
        "title_font": {"color": "#111827", "size": 13},
    }
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def overview_page(artifacts: dict[str, object]) -> None:
    final_metrics = artifacts["final_metrics"]
    dataset_summary = artifacts["dataset_summary"]
    decisions = artifacts["decisions"]
    model_comparison = artifacts["model_comparison"]

    default = final_metrics["default_threshold_metrics"]
    accuracy_optimal = final_metrics["accuracy_optimal_threshold_metrics"]
    operational = final_metrics["operational_threshold_metrics"]
    threshold = final_metrics["threshold_summary"]["cost_sensitive_threshold"]
    intervention_count = int(decisions["intervention_candidate"].sum())

    section_heading("Portfolio Overview", "Hold-out performance and operational retention load.")
    render_metric_cards(
        [
            ("Accuracy", fmt_num(default["accuracy"]), "Standard 0.50 threshold"),
            ("Best accuracy", fmt_num(accuracy_optimal["accuracy"]), "Threshold-optimized score"),
            ("PR-AUC", fmt_num(default["pr_auc"]), "Minority-class ranking quality"),
            ("Recall", fmt_num(operational["recall"]), f"At threshold {threshold:.2f}"),
            ("Interventions", f"{intervention_count:,}", "Flagged by threshold"),
        ]
    )

    section_heading("Risk and Model Signal")
    chart_cols = st.columns([1, 1])
    risk_counts = decisions["risk_tier"].value_counts().reindex(["low", "medium", "high"]).reset_index()
    risk_counts.columns = ["risk_tier", "customers"]
    risk_fig = px.bar(
        risk_counts,
        x="risk_tier",
        y="customers",
        color="risk_tier",
        color_discrete_map={"low": "#178a52", "medium": "#b7791f", "high": "#c2412d"},
        title="Operational risk tier distribution",
        labels={"risk_tier": "", "customers": "Customers"},
    )
    chart_cols[0].plotly_chart(style_fig(risk_fig), use_container_width=True, config=CHART_CONFIG)

    top_models = model_comparison.sort_values("cv_pr_auc_mean", ascending=False).head(7)
    model_fig = px.bar(
        top_models.sort_values("cv_pr_auc_mean", ascending=True),
        x="cv_pr_auc_mean",
        y="model",
        orientation="h",
        color="cv_pr_auc_mean",
        color_continuous_scale=["#dbeafe", "#1f6feb"],
        title="Cross-validated PR-AUC by model",
        labels={"cv_pr_auc_mean": "CV PR-AUC", "model": ""},
    )
    model_fig.update_layout(coloraxis_showscale=False)
    chart_cols[1].plotly_chart(style_fig(model_fig), use_container_width=True, config=CHART_CONFIG)

    section_heading("Dataset Footprint")
    render_metric_cards(
        [
            ("Rows", f"{dataset_summary['rows']:,}", "IBM Telco customers"),
            ("Churn rate", fmt_pct(dataset_summary["churn_rate"], 1), "Observed target rate"),
            ("Training rows", f"{dataset_summary['train_rows']:,}", "Stratified sample"),
            ("Test rows", f"{dataset_summary['test_rows']:,}", "Final hold-out sample"),
        ]
    )


def performance_page(artifacts: dict[str, object]) -> None:
    model_comparison = artifacts["model_comparison"]
    thresholds = artifacts["thresholds"]
    tenure = artifacts["tenure"]
    final_metrics = artifacts["final_metrics"]

    section_heading("Model Evaluation", "PR-AUC is treated as the primary comparison metric.")
    visible_columns = [
        "model",
        "cv_accuracy_mean",
        "cv_pr_auc_mean",
        "cv_roc_auc_mean",
        "cv_recall_mean",
        "cv_f1_mean",
        "test_accuracy",
        "test_pr_auc",
        "test_roc_auc",
        "test_recall",
        "test_f1",
    ]
    display_table = model_comparison[visible_columns].sort_values("cv_pr_auc_mean", ascending=False)
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
        height=300,
        column_config={
            "cv_accuracy_mean": st.column_config.NumberColumn("CV Accuracy", format="%.3f"),
            "cv_pr_auc_mean": st.column_config.NumberColumn("CV PR-AUC", format="%.3f"),
            "cv_roc_auc_mean": st.column_config.NumberColumn("CV ROC-AUC", format="%.3f"),
            "cv_recall_mean": st.column_config.NumberColumn("CV Recall", format="%.3f"),
            "cv_f1_mean": st.column_config.NumberColumn("CV F1", format="%.3f"),
            "test_accuracy": st.column_config.NumberColumn("Test Accuracy", format="%.3f"),
            "test_pr_auc": st.column_config.NumberColumn("Test PR-AUC", format="%.3f"),
            "test_roc_auc": st.column_config.NumberColumn("Test ROC-AUC", format="%.3f"),
            "test_recall": st.column_config.NumberColumn("Test Recall", format="%.3f"),
            "test_f1": st.column_config.NumberColumn("Test F1", format="%.3f"),
        },
    )

    threshold_cols = st.columns([1, 1])
    threshold_fig = px.line(
        thresholds,
        x="threshold",
        y=["precision", "recall", "f1"],
        color_discrete_sequence=["#1f6feb", "#178a52", "#b7791f"],
        title="Threshold tradeoff",
        labels={"value": "Score", "variable": "Metric"},
    )
    threshold_cols[0].plotly_chart(style_fig(threshold_fig), use_container_width=True, config=CHART_CONFIG)

    cost_fig = px.line(
        thresholds,
        x="threshold",
        y="business_cost",
        title="Cost-sensitive threshold analysis",
        labels={"business_cost": "Business cost"},
    )
    selected_threshold = final_metrics["threshold_summary"]["cost_sensitive_threshold"]
    cost_fig.add_vline(x=selected_threshold, line_width=2, line_dash="dash", line_color="#c2412d")
    threshold_cols[1].plotly_chart(style_fig(cost_fig), use_container_width=True, config=CHART_CONFIG)

    section_heading(
        "Lifecycle-Oriented Tenure Analysis",
        "Tenure bands are a temporal proxy because the source data is cross-sectional.",
    )
    tenure_fig = px.bar(
        tenure,
        x="tenure_band",
        y="churn_rate",
        text=tenure["churn_rate"].map(lambda value: f"{value:.1%}"),
        color="churn_rate",
        color_continuous_scale=["#d1fae5", "#0f766e"],
        title="Observed churn rate by tenure band",
        labels={"tenure_band": "", "churn_rate": "Observed churn rate"},
    )
    tenure_fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(tenure_fig, height=400), use_container_width=True, config=CHART_CONFIG)


def customer_page(artifacts: dict[str, object]) -> None:
    decisions = artifacts["decisions"]
    shap_local = artifacts["shap_local"]
    lime_local = artifacts["lime_local"]

    section_heading("Customer Desk", "Prioritize retention work from risk, drivers, and action fit.")
    filter_cols = st.columns([1, 1, 2])
    tier_filter = filter_cols[0].selectbox("Risk tier", ["all", "high", "medium", "low"], index=1)
    only_intervention = filter_cols[1].checkbox("Intervention candidates only", value=True)

    filtered = decisions.copy()
    if tier_filter != "all":
        filtered = filtered[filtered["risk_tier"] == tier_filter]
    if only_intervention:
        filtered = filtered[filtered["intervention_candidate"]]
    filtered = filtered.sort_values("churn_probability", ascending=False)

    if filtered.empty:
        st.warning("No customers match the selected filters.")
        return

    customer_id = filter_cols[2].selectbox("Customer", filtered["customerID"].tolist())
    customer = filtered[filtered["customerID"] == customer_id].iloc[0]
    tier = str(customer["risk_tier"])
    probability = float(customer["churn_probability"])

    st.markdown(
        f'<span class="risk-pill" style="background:{risk_color(tier)}">{escape(tier)} risk</span>',
        unsafe_allow_html=True,
    )
    render_metric_cards(
        [
            ("Churn probability", fmt_pct(probability, 1), "Model probability"),
            ("Actual label", "Churn" if int(customer["Churn"]) == 1 else "No churn", "Hold-out outcome"),
            ("Recommended action", str(customer["recommended_action"]), "Primary retention move"),
            ("Monthly charges", f"${float(customer['MonthlyCharges']):.2f}", "Current account value"),
        ]
    )

    detail_cols = st.columns([1, 1])
    profile_fields = [
        "Contract",
        "tenure",
        "tenure_band",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "PaymentMethod",
        "total_active_services",
    ]
    profile_items = "".join(
        (
            '<div class="profile-item">'
            f'<div class="profile-label">{escape(field)}</div>'
            f'<div class="profile-value">{escape(str(customer[field]))}</div>'
            "</div>"
        )
        for field in profile_fields
    )
    detail_cols[0].markdown(
        dedent(
            f"""
        <div class="section-heading"><h2 class="section-title">Customer profile</h2></div>
        <div class="profile-grid">{profile_items}</div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )

    detail_cols[1].markdown(
        dedent(
            f"""
        <div class="section-heading"><h2 class="section-title">Retention recommendation</h2></div>
        <div class="recommendation-box">{escape(str(customer["recommendation_rationale"]))}</div>
        """
        ).strip(),
        unsafe_allow_html=True,
    )

    shap_row = shap_local[shap_local["customerID"] == customer_id]
    if not shap_row.empty:
        drivers = str(shap_row.iloc[0]["top_shap_drivers"]).split("; ")
        driver_html = "".join(f'<div class="driver-item">{escape(driver)}</div>' for driver in drivers)
        st.markdown(
            dedent(
                f"""
            <div class="section-heading"><h2 class="section-title">Top local SHAP drivers</h2></div>
            <div class="driver-list">{driver_html}</div>
            """
            ).strip(),
            unsafe_allow_html=True,
        )
    else:
        st.caption("This customer is outside the saved SHAP sample; use the global driver view for aggregate explanation.")

    lime_row = lime_local[lime_local["customerID"] == customer_id]
    if not lime_row.empty:
        with st.expander("LIME local explanation"):
            st.write(str(lime_row.iloc[0]["lime_explanation"]))

    section_heading("Filtered Customer Queue")
    queue = filtered[
        [
            "customerID",
            "churn_probability",
            "risk_tier",
            "intervention_candidate",
            "recommended_action",
            "Contract",
            "tenure",
            "MonthlyCharges",
        ]
    ]
    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True,
        height=340,
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn probability",
                min_value=0,
                max_value=1,
                format="%.2f",
            ),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
        },
    )


def explanations_page(artifacts: dict[str, object]) -> None:
    shap_df = artifacts["shap"]
    decisions = artifacts["decisions"]

    section_heading("Drivers & Actions", "Global explanation and retention workload composition.")
    top_n = st.slider("Number of SHAP features", min_value=5, max_value=25, value=15)
    top_features = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)
    shap_fig = px.bar(
        top_features,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale=["#dbeafe", "#1f6feb"],
        title="Global SHAP feature importance",
        labels={"mean_abs_shap": "Mean absolute SHAP", "feature": ""},
    )
    shap_fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style_fig(shap_fig, height=510), use_container_width=True, config=CHART_CONFIG)

    chart_cols = st.columns([1, 1])
    action_counts = decisions["recommended_action"].value_counts().reset_index()
    action_counts.columns = ["recommended_action", "customers"]
    action_fig = px.bar(
        action_counts.sort_values("customers", ascending=True),
        x="customers",
        y="recommended_action",
        orientation="h",
        color="customers",
        color_continuous_scale=["#e0f2fe", "#0f766e"],
        title="Recommended retention actions",
        labels={"recommended_action": "", "customers": "Customers"},
    )
    action_fig.update_layout(coloraxis_showscale=False)
    chart_cols[0].plotly_chart(style_fig(action_fig, height=390), use_container_width=True, config=CHART_CONFIG)

    tier_action = (
        decisions.groupby(["risk_tier", "recommended_action"], observed=False)
        .size()
        .reset_index(name="customers")
    )
    mix_fig = px.bar(
        tier_action,
        x="risk_tier",
        y="customers",
        color="recommended_action",
        title="Action mix by risk tier",
        labels={"risk_tier": "", "customers": "Customers", "recommended_action": "Action"},
    )
    chart_cols[1].plotly_chart(style_fig(mix_fig, height=390), use_container_width=True, config=CHART_CONFIG)

    st.dataframe(
        shap_df.head(12),
        use_container_width=True,
        hide_index=True,
        height=330,
        column_config={
            "mean_abs_shap": st.column_config.NumberColumn("Mean absolute SHAP", format="%.3f"),
        },
    )


def main() -> None:
    configure_page()
    artifacts = load_artifacts()
    if "missing" in artifacts:
        render_missing(artifacts["missing"])
        return

    render_shell_header(artifacts)
    selected_page = render_navigation()
    pages = {
        "Overview": overview_page,
        "Model Performance": performance_page,
        "Customer Desk": customer_page,
        "Drivers & Actions": explanations_page,
    }
    pages[selected_page](artifacts)


if __name__ == "__main__":
    main()
