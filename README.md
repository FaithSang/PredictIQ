# PredictIQ  Agentic Predictive Maintenance Copilot

Built for **ABB Accelerator 2026  Theme 1: Agentic Predictive Maintenance Studio**

PredictIQ turns raw turbofan sensor data into explainable, actionable maintenance
recommendations: it predicts Remaining Useful Life (RUL), explains *why* an engine is
flagged using SHAP, and translates that into plain-English guidance via the Claude API.

## What it does
1. **Predicts** Remaining Useful Life for each engine using XGBoost, trained on NASA's
   C-MAPSS FD002 dataset (260 engines, 6 operating conditions, a harder, more realistic
   variant than the commonly-used FD001)
2. **Compares** three models (Random Forest, XGBoost, LightGBM) on accuracy and training
   time, so the model choice is a defensible decision, not an assumption
3. **Explains** every prediction using SHAP, both risk-increasing and protective sensor
   readings, translated from raw sensor IDs into real physical names (e.g. `s13` →
   "Corrected fan speed")
4. **Advises**: the Claude API converts the structured SHAP output into a 2-3 sentence,
   technician-ready recommendation with concrete urgency, the "agentic" layer the
   challenge brief explicitly asks for

## Results
| Model | Test RMSE (cycles) | Test MAE (cycles) | Train time |
|---|---|---|---|
| LightGBM | 16.99 | 12.84 | ~1s |
| XGBoost | 17.13 | 12.92 | ~1.5s |
| Random Forest | 18.13 | 13.69 | ~68s |

XGBoost was selected as the production model despite LightGBM's marginal edge, because
its SHAP `TreeExplainer` integration is faster and more mature, prioritizing
explainability quality over a ~0.1-cycle RMSE difference.

## Quick start
See [INSTRUCTIONS.md](./INSTRUCTIONS.md) for full setup and run instructions.

```
pip install -r requirements.txt
python data_utils.py
streamlit run app.py
```

## Tech stack
Python, XGBoost, LightGBM, scikit-learn, SHAP, Streamlit, Claude API (Anthropic)

## Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation, FD002 subset.
