"""
PredictIQ - Day 1: Baseline Random Forest, split by engine (not by row) to avoid leakage.
Goal: prove ingest -> features -> train -> predict -> evaluate works end to end.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURE_COLS = ["set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]

def load():
    train = pd.read_csv("data/train_labeled.csv")
    test = pd.read_csv("data/test_labeled.csv")
    return train, test

def split_by_engine(train, test_size=0.2, seed=42):
    """Hold out whole engines for validation, never split a single engine's
    cycles across train/val -- that would leak future cycles into training."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_tr, idx_val = next(gss.split(train, groups=train["unit"]))
    return train.iloc[idx_tr], train.iloc[idx_val]

def evaluate(y_true, y_pred, label):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{label}] RMSE={rmse:.2f} cycles | MAE={mae:.2f} cycles")
    return {"rmse": rmse, "mae": mae}

if __name__ == "__main__":
    train, test = load()
    tr, val = split_by_engine(train)

    X_tr, y_tr = tr[FEATURE_COLS], tr["RUL"]
    X_val, y_val = val[FEATURE_COLS], val["RUL"]
    X_test, y_test = test[FEATURE_COLS], test["RUL"]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
    )
    model.fit(X_tr, y_tr)

    print(f"Train engines: {tr['unit'].nunique()} | Val engines: {val['unit'].nunique()}")
    evaluate(y_val, model.predict(X_val), "VALIDATION (held-out engines)")

    # Real test-set score: engines end at an arbitrary cycle before failure, and we
    # score against the true RUL at that cutoff -- the actual C-MAPSS benchmark setup.
    test_last_cycle = test.loc[test.groupby("unit")["cycle"].idxmax()]
    evaluate(
        test_last_cycle["RUL"],
        model.predict(test_last_cycle[FEATURE_COLS]),
        "TEST (last cycle per engine, true benchmark)",
    )

    print("\nPipeline check: OK -- ingest, features, train, predict, evaluate all run.")
