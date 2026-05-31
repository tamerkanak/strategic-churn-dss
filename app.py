"""Deployment entrypoint for the churn DSS Streamlit app."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_dashboard_main():
    dashboard_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("churn_dss_streamlit_app", dashboard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Streamlit dashboard from {dashboard_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


main = _load_dashboard_main()

if __name__ == "__main__":
    main()
