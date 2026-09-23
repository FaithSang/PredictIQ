# PredictIQ Instructions to Run

## Setup
1. Clone the repository: `git clone <your-repo-url>` and `cd predictiq`
2. (Recommended) create a virtual environment: `python -m venv .venv` then activate it
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure the `data/` folder contains: `train_FD002.txt`, `test_FD002.txt`, `RUL_FD002.txt`
   (NASA C-MAPSS Turbofan Degradation dataset, FD002 subset)

## Prepare labeled data (one-time, generates data/train_labeled.csv and test_labeled.csv)
```
python data_utils.py
```

## Run the dashboard
```
streamlit run app.py
```
This opens the dashboard in your browser (usually http://localhost:8501).

Theming lives in `.streamlit/config.toml` — make sure that folder ships with the repo
(it's easy to miss since the leading dot hides it in some file browsers).

## Using the dashboard
The app walks through three steps:

1. **Choose your data** - use the built-in demo fleet (259 NASA test engines), or upload
   your own CSV. A "Download template CSV" button shows the exact 26-column format
   expected (`unit`, `cycle`, `set1`-`set3`, `s1`-`s21`). Both a full multi-machine fleet
   log and a single machine's readings are supported.
2. **Fleet risk at a glance** - a color-coded table of every machine's estimated cycles
   remaining, risk %, and status (HIGH / MEDIUM / LOW), with counts summarized at the top.
3. **Inspect a machine** - pick any machine to see its SHAP-based risk/protective factor
   chart (real sensor names, not raw codes) and a plain-English recommendation.

By default, recommendations run in **offline preview mode** (no API cost, no key
required). To generate live Claude API recommendations:
- Set the `ANTHROPIC_API_KEY` environment variable
  - PowerShell: `$env:ANTHROPIC_API_KEY="your-key-here"`
  - macOS/Linux: `export ANTHROPIC_API_KEY="your-key-here"`
- Toggle "Use live Claude API" next to the selected machine's recommendation.
- If the API call fails (e.g. low credits, invalid key), the app shows a clear error
  message and automatically falls back to the offline preview rather than crashing.

## Deploying (e.g. Streamlit Community Cloud)
Add `ANTHROPIC_API_KEY` under the app's Secrets panel (TOML format) instead of a local
environment variable — the app checks `st.secrets` automatically as a fallback.

## Optional: rerun model comparison / SHAP sanity checks standalone
```
python baseline_model.py      # baseline Random Forest, pipeline check
python compare_models.py      # Random Forest vs XGBoost vs LightGBM comparison
python shap_explain.py        # SHAP explanations for highest/lowest risk engines
python advisor.py --mock      # Claude advisory layer, offline test mode
```

## Project structure
```
predictiq/
├── .streamlit/
│   └── config.toml         # dashboard theme
├── data/                    # raw NASA files + generated labeled CSVs
├── data_utils.py            # data loading, profiling, RUL labeling
├── baseline_model.py         # baseline Random Forest + engine-level train/val split
├── compare_models.py         # RF vs XGBoost vs LightGBM comparison
├── shap_explain.py           # SHAP explainability (risk + protective factors)
├── sensor_names.py           # sensor ID -> human-readable name mapping
├── advisor.py                # Claude API plain-English recommendation layer
├── app.py                    # Streamlit dashboard (main entry point)
└── requirements.txt
```
