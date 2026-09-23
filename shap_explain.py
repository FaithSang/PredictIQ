"""
PredictIQ - Day 3: SHAP explainability on XGBoost.
Produces per-engine "why is this engine flagged" breakdowns using SHAP TreeExplainer,
then sanity-checks the output against known high-risk and low-risk engines.
"""
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

from baseline_model import load, split_by_engine, FEATURE_COLS
from data_utils import risk_tier
from sensor_names import describe

RUL_CAP = 125


def train_xgb(X_tr, y_tr):
    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)
    return model


def explain_engine(model, explainer, X_row, engine_id, predicted_rul, top_n=3):
    """Return a structured explanation dict for one engine's current reading,
    matching the format the Claude API layer (Day 4) will consume.
    Includes both risk-increasing factors (most negative SHAP) and protective
    factors (most positive SHAP) so a healthy engine's explanation doesn't read
    as falsely alarming when only the negative tail is shown."""
    shap_values = explainer.shap_values(X_row.astype(float))[0]  # single row
    base_value = explainer.expected_value

    contrib = pd.Series(shap_values, index=FEATURE_COLS)
    # Negative SHAP = pushes RUL down = increases risk. Sort by magnitude of
    # risk-increasing contribution (most negative first).
    risk_drivers = contrib.sort_values().head(top_n)
    # Positive SHAP = pushes RUL up = protective / reassuring reading.
    protective_drivers = contrib.sort_values(ascending=False).head(top_n)

    total_negative = contrib[contrib < 0].sum()
    total_positive = contrib[contrib > 0].sum()

    def _build_factors(series, total):
        out = []
        for feat, val in series.items():
            pct = (val / total * 100) if total != 0 else 0
            out.append({
                "feature": feat,
                "feature_name": describe(feat),
                "shap_value": round(float(val), 2),
                "pct_contribution": round(float(pct), 1),
                "raw_reading": round(float(X_row[feat].values[0]), 2),
            })
        return out

    risk_factors = _build_factors(risk_drivers, total_negative)
    protective_factors = _build_factors(protective_drivers, total_positive)

    risk_pct = round(100 * (1 - predicted_rul / RUL_CAP), 1)
    risk_pct = max(0, min(100, risk_pct))

    return {
        "engine_id": int(engine_id),
        "predicted_rul": round(float(predicted_rul), 1),
        "risk_score_pct": risk_pct,
        "risk_tier": risk_tier(predicted_rul),
        "base_value": round(float(base_value), 1),
        "net_negative_contribution": round(float(total_negative), 1),
        "net_positive_contribution": round(float(total_positive), 1),
        "top_risk_factors": risk_factors,
        "top_protective_factors": protective_factors,
    }


def print_explanation(exp):
    print(f"Engine {exp['engine_id']}: RUL={exp['predicted_rul']} cycles | "
          f"Risk={exp['risk_score_pct']}% ({exp['risk_tier']})")
    print(f"  Risk-increasing factors (net {exp['net_negative_contribution']}):")
    for f in exp["top_risk_factors"]:
        print(f"    {f['feature_name']}: reading={f['raw_reading']} | "
              f"SHAP={f['shap_value']} | {f['pct_contribution']}% of risk pull")
    print(f"  Protective factors (net +{exp['net_positive_contribution']}):")
    for f in exp["top_protective_factors"]:
        print(f"    {f['feature_name']}: reading={f['raw_reading']} | "
              f"SHAP={f['shap_value']} | {f['pct_contribution']}% of protective pull")


if __name__ == "__main__":
    train, test = load()
    tr, val = split_by_engine(train)
    X_tr, y_tr = tr[FEATURE_COLS], tr["RUL"]

    model = train_xgb(X_tr, y_tr)
    explainer = shap.TreeExplainer(model)

    # Sanity check: pick the true highest-risk and lowest-risk engines from the
    # test set's last-cycle rows, and confirm the SHAP explanation direction
    # makes sense (high risk -> factors pushing RUL down).
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
    preds = model.predict(test_last[FEATURE_COLS])
    test_last["pred_rul"] = preds

    highest_risk = test_last.nsmallest(3, "pred_rul")
    lowest_risk = test_last.nlargest(3, "pred_rul")

    print("=== SANITY CHECK: 3 HIGHEST-RISK ENGINES ===")
    for _, row in highest_risk.iterrows():
        X_row = row[FEATURE_COLS].to_frame().T
        exp = explain_engine(model, explainer, X_row, row["unit"], row["pred_rul"])
        print_explanation(exp)
        print()

    print("=== SANITY CHECK: 3 LOWEST-RISK (HEALTHY) ENGINES ===")
    for _, row in lowest_risk.iterrows():
        X_row = row[FEATURE_COLS].to_frame().T
        exp = explain_engine(model, explainer, X_row, row["unit"], row["pred_rul"])
        print_explanation(exp)
        print()

    print("Pipeline check: SHAP explanations generated for both risk extremes.")
