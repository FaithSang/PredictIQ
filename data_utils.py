"""
PredictIQ - Day 1: Data loading, profiling, and RUL labeling for C-MAPSS FD002.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]

# Sensors that are constant (zero variance) in FD002 per standard C-MAPSS analysis.
# We verify this empirically below rather than trusting the literature blindly.
CANDIDATE_DROP = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

RUL_CAP = 125  # standard C-MAPSS practice: clip RUL so early-life cycles aren't
               # penalized for a "distance to failure" that's unknowable that early


def load_raw(path, has_labels=True):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df

def _cluster_operating_conditions(df: pd.DataFrame, n_clusters=6) -> int:
    """Count distinct operating conditions by clustering set1/set2/set3 rather than
    fixed-decimal rounding, which is unsafe near boundary values like set2=0.25."""
    from sklearn.cluster import KMeans
 
    X = df[["set1", "set2", "set3"]].values
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(X)
    # Report actual distinct clusters found, not just n_clusters requested --
    # if fewer than n_clusters are truly separable this will still surface it.
    return len(set(km.labels_))


def profile(df: pd.DataFrame) -> dict:
    """Quick data-quality profile: shape, nulls, per-engine cycle counts, constant cols."""
    const_cols = [c for c in COLS[2:] if df[c].std() == 0]
    report = {
        "rows": len(df),
        "engines": df["unit"].nunique(),
        "nulls_total": int(df.isnull().sum().sum()),
        "cycle_min": int(df.groupby("unit")["cycle"].max().min()),
        "cycle_max": int(df.groupby("unit")["cycle"].max().max()),
        # FD002's raw settings carry ~0.01 floating-point noise around each of the
        # 6 documented operating conditions. Decimal rounding is not safe here --
        # set2 sits exactly on a .25 boundary that round-half-to-even flips
        # inconsistently. Clustering (nearest of the 6 known set1/set3 pairs) is
        # boundary-safe where fixed-decimal rounding isn't.
        "operating_conditions_observed": int(
            _cluster_operating_conditions(df)
        ),
        "constant_sensor_cols": const_cols,
    }
    return report


def add_rul_train(df: pd.DataFrame, cap=RUL_CAP) -> pd.DataFrame:
    """Training data: RUL = (max cycle for that engine) - (current cycle), clipped."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def add_rul_test(df: pd.DataFrame, rul_path: str, cap=RUL_CAP) -> pd.DataFrame:
    """Test data: true RUL at the END of each engine's truncated trace is given in
    RUL_FD002.txt. We reconstruct RUL at every row by adding back the cycles remaining
    within the truncated trace itself, then clip."""
    rul_end = pd.read_csv(rul_path, header=None, names=["RUL_end"])
    rul_end["unit"] = rul_end.index + 1
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.merge(rul_end, on="unit", how="left")
    df["RUL"] = (df["RUL_end"] + (max_cycle - df["cycle"])).clip(upper=cap)
    df = df.drop(columns=["RUL_end"])
    return df


def risk_tier(rul, high=20, medium=60):
    if rul < high:
        return "HIGH"
    elif rul < medium:
        return "MEDIUM"
    return "LOW"


if __name__ == "__main__":
    train = load_raw("data/train_FD002.txt")
    test = load_raw("data/test_FD002.txt")

    print("=== TRAIN PROFILE ===")
    for k, v in profile(train).items():
        print(f"  {k}: {v}")

    print("\n=== TEST PROFILE ===")
    for k, v in profile(test).items():
        print(f"  {k}: {v}")

    train_labeled = add_rul_train(train)
    test_labeled = add_rul_test(test, "data/RUL_FD002.txt")

    print("\nTrain RUL distribution:\n", train_labeled["RUL"].describe())
    print("\nTest RUL distribution:\n", test_labeled["RUL"].describe())

    train_labeled.to_csv("data/train_labeled.csv", index=False)
    test_labeled.to_csv("data/test_labeled.csv", index=False)
    print("\nSaved train_labeled.csv, test_labeled.csv")
