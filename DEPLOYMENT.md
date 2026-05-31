# Deployment Guide

This project is a Python Streamlit application, so it should be deployed to a platform that can run a long-lived Python web process. Vercel and Netlify are optimized for static/frontend sites and serverless functions; they are not the best fit for this app.

## Recommended: Streamlit Community Cloud

Use this when you want the fastest deployment with the least configuration.

1. Create a GitHub repository for this project folder.
2. Push these files to GitHub, including:
   - `app.py`
   - `app/`
   - `src/`
   - `artifacts/`
   - `data/`
   - `reports/`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Open `https://share.streamlit.io`.
4. Sign in with GitHub.
5. Click **Create app**.
6. Select the repository and branch.
7. Set the main file path to:

   ```text
   app.py
   ```

8. Deploy.

The app should start with:

```bash
streamlit run app.py
```

No secrets or environment variables are required.

## Alternative: Hugging Face Spaces

Use this if you want a public ML/demo page that feels like a hosted model demo.

1. Create a new Space at `https://huggingface.co/new-space`.
2. Select **Streamlit** as the SDK.
3. Upload or push the same project files.
4. Ensure the Space has:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `artifacts/`
   - `src/`
   - `app/`
5. The Space will run `app.py` automatically.

## Alternative: Render

Use this if you prefer a general web service platform.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Pre-Deploy Checklist

Run locally before deploying:

```powershell
py -m ruff check .
py -m pytest
py -m streamlit run app.py
```

Then open:

```text
http://127.0.0.1:8501
```

If the app reports missing artifacts, run:

```powershell
py -m src.churn_dss.train --data-source auto --random-state 42
py -m src.churn_dss.generate_report
```
