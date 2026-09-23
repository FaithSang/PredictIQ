"""
PredictIQ - Day 2: Model comparison -- Random Forest vs XGBoost vs LightGBM.
Reuses the same engine-level split and evaluation from baseline_model.py so the
comparison is apples-to-apples (same train/val engines, same features, same metric).
"""
import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from baseline_model import load, split_by_engine, evaluate, FEATURE_COLS

MODELS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
        verbose=-1
    ),
}


def run_comparison():
    train, test = load()
    tr, val = split_by_engine(train)

    X_tr, y_tr = tr[FEATURE_COLS], tr["RUL"]
    X_val, y_val = val[FEATURE_COLS], val["RUL"]

    # Score the same way as baseline_model.py: last cycle per test engine,
    # which is the real C-MAPSS benchmark convention.
    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()]
    X_test, y_test = test_last[FEATURE_COLS], test_last["RUL"]

    results = []
    fitted = {}
    for name, model in MODELS.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        fitted[name] = model

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate(y_val, val_pred, f"{name} - VALIDATION")
        test_metrics = evaluate(y_test, test_pred, f"{name} - TEST")

        results.append({
            "model": name,
            "val_rmse": round(val_metrics["rmse"], 2),
            "val_mae": round(val_metrics["mae"], 2),
            "test_rmse": round(test_metrics["rmse"], 2),
            "test_mae": round(test_metrics["mae"], 2),
            "train_time_sec": round(train_time, 2),
        })
        print()

    comparison = pd.DataFrame(results).sort_values("test_rmse")
    print("=== MODEL COMPARISON (sorted by test RMSE, lower is better) ===")
    print(comparison.to_string(index=False))

    comparison.to_csv("model_comparison.csv", index=False)
    print("\nSaved model_comparison.csv")

    return comparison, fitted


if __name__ == "__main__":
    comparison, fitted = run_comparison()
