# PredictIQ — Instructions to Run

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

## Using the dashboard
- The fleet overview table on the left lists all engines with predicted Remaining
  Useful Life (RUL), risk %, and risk tier (HIGH / MEDIUM / LOW).
- Filter by tier using the sidebar, or select any engine from the dropdown to inspect it.
- The right panel shows the engine's SHAP risk/protective factor chart and a plain-English
  maintenance recommendation.
- By default, recommendations run in **mock mode** (no API cost, no key required) so the
  full pipeline can be tested for free.
- To generate live Claude API recommendations: set the `ANTHROPIC_API_KEY` environment
  variable, then toggle "Use live Claude API" in the sidebar.
  - PowerShell: `$env:ANTHROPIC_API_KEY="your-key-here"`
  - macOS/Linux: `export ANTHROPIC_API_KEY="your-key-here"`

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
├── data/                  # raw NASA files + generated labeled CSVs
├── .streamlit/            #config.toml files
├── data_utils.py          # data loading, profiling, RUL labeling
├── baseline_model.py       # baseline Random Forest + engine-level train/val split
├── compare_models.py       # RF vs XGBoost vs LightGBM comparison
├── shap_explain.py         # SHAP explainability (risk + protective factors)
├── sensor_names.py         # sensor ID -> human-readable name mapping
├── advisor.py               # Claude API plain-English recommendation layer
├── app.py                   # Streamlit dashboard (main entry point)
└── requirements.txt
├── README.md
```
