"""
PredictIQ - Streamlit dashboard.
Run with: streamlit run app.py

Three-step flow: choose data -> see fleet risk at a glance -> inspect one engine.
"""
import os
import io
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from baseline_model import load, split_by_engine, FEATURE_COLS
from shap_explain import train_xgb, explain_engine
from advisor import generate_recommendation, AdvisorError

st.set_page_config(page_title="PredictIQ", page_icon="\u2699\ufe0f", layout="wide")

# Bridge Streamlit Cloud's secrets manager to the environment variable advisor.py
# expects. Locally, os.environ already has the key (if set) and this is a no-op.
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

TIER_COLOR = {"HIGH": "#E5484D", "MEDIUM": "#E8A33D", "LOW": "#3DD68C"}
ALL_COLS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]

CUSTOM_CSS = """
<style>
.badge {
    display: inline-block; padding: 3px 12px; border-radius: 3px;
    font-weight: 600; font-size: 0.85rem; letter-spacing: 0.02em;
}
.step-label {
    color: #8B94A3; font-size: 0.85rem; margin-bottom: -0.5rem;
}
</style>
"""


@st.cache_resource
def get_model_and_explainer():
    train, _ = load()
    tr, _ = split_by_engine(train)
    model = train_xgb(tr[FEATURE_COLS], tr["RUL"])
    explainer = shap.TreeExplainer(model)
    return model, explainer


def score_fleet(_model, df):
    """Take a raw dataframe (unit, cycle, features...) and return one scored row
    per engine (its most recent cycle), with predicted RUL, risk %, and tier."""
    last = df.loc[df.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
    last["pred_rul"] = _model.predict(last[FEATURE_COLS].astype(float))
    last["risk_pct"] = (100 * (1 - last["pred_rul"] / 125)).clip(0, 100).round(1)
    last["risk_tier"] = last["pred_rul"].apply(
        lambda r: "HIGH" if r < 20 else ("MEDIUM" if r < 60 else "LOW")
    )
    return last.sort_values("pred_rul")


def badge(tier):
    return f'<span class="badge" style="background:{TIER_COLOR[tier]}22; color:{TIER_COLOR[tier]};">{tier}</span>'


def plot_factors(exp):
    risk = [(f["feature_name"], f["shap_value"]) for f in exp["top_risk_factors"]]
    protective = [(f["feature_name"], f["shap_value"]) for f in exp["top_protective_factors"]]
    all_factors = sorted(risk + protective, key=lambda x: x[1])

    labels = [f[0] for f in all_factors]
    values = [f[1] for f in all_factors]
    colors = [TIER_COLOR["HIGH"] if v < 0 else TIER_COLOR["LOW"] for v in values]

    plt.rcParams.update({
        "figure.facecolor": "#1B222B", "axes.facecolor": "#1B222B",
        "text.color": "#E7EAEE", "axes.labelcolor": "#E7EAEE",
        "xtick.color": "#E7EAEE", "ytick.color": "#E7EAEE",
        "axes.edgecolor": "#3A4250",
    })
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#8B94A3", linewidth=0.8)
    ax.set_xlabel("Pulls risk down \u2190          \u2192 pulls risk up")
    fig.tight_layout()
    return fig


def make_template_csv():
    demo_row = {c: 0 for c in ALL_COLS}
    demo_row.update({"unit": 1, "cycle": 1, "set1": 20.0, "set2": 0.7, "set3": 100.0})
    for i in range(1, 22):
        demo_row[f"s{i}"] = 500.0
    return pd.DataFrame([demo_row]).to_csv(index=False)


def try_parse_upload(uploaded_file):
    """Accepts a CSV with a header row matching ALL_COLS. Returns (df, error)."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"Couldn't read that file as a CSV: {e}"

    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        return None, (
            f"Missing {len(missing)} expected column(s): {', '.join(missing[:6])}"
            + ("..." if len(missing) > 6 else "")
            + ". Download the template below to see the exact format needed."
        )
    try:
        df[ALL_COLS] = df[ALL_COLS].astype(float)
        df["unit"] = df["unit"].astype(int)
        df["cycle"] = df["cycle"].astype(int)
    except Exception as e:
        return None, f"Some values couldn't be read as numbers: {e}"
    return df[ALL_COLS], None


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("PredictIQ")
    st.caption(
        "Upload sensor readings from your equipment, and PredictIQ estimates how much "
        "life each machine has left, explains why, and tells you what to do about it."
    )

    model, explainer = get_model_and_explainer()

    # ---------- Step 1: data source ----------
    st.markdown('<p class="step-label">STEP 1</p>', unsafe_allow_html=True)
    st.subheader("Choose your data")
    source = st.radio(
        "Data source", ["Use the demo fleet", "Upload my own CSV"],
        label_visibility="collapsed", horizontal=True,
    )

    if source == "Use the demo fleet":
        _, test = load()
        raw = test[ALL_COLS]
        st.caption("Showing 259 turbofan engines from NASA's public test dataset.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            uploaded = st.file_uploader(
                "Upload a CSV of sensor readings (one row per machine per reading cycle)",
                type=["csv"],
            )
        with col_b:
            st.download_button(
                "Download template CSV", make_template_csv(),
                file_name="predictiq_template.csv", mime="text/csv",
            )
            st.caption("Same 26-column format PredictIQ expects.")

        if uploaded is None:
            st.info("Upload a CSV to continue, or switch to the demo fleet above.")
            return
        raw, error = try_parse_upload(uploaded)
        if error:
            st.error(error)
            return
        st.success(f"Loaded {raw['unit'].nunique()} machine(s), {len(raw)} reading(s).")

    fleet = score_fleet(model, raw)

    # ---------- Step 2: fleet overview ----------
    st.divider()
    st.markdown('<p class="step-label">STEP 2</p>', unsafe_allow_html=True)
    st.subheader("Fleet risk at a glance")

    counts = fleet["risk_tier"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("High risk", int(counts.get("HIGH", 0)))
    c2.metric("Medium risk", int(counts.get("MEDIUM", 0)))
    c3.metric("Low risk", int(counts.get("LOW", 0)))

    display_df = fleet[["unit", "pred_rul", "risk_pct", "risk_tier"]].rename(
        columns={"unit": "Machine", "pred_rul": "Cycles left", "risk_pct": "Risk %"}
    )
    st.dataframe(
        display_df,
        column_config={
            "Risk %": st.column_config.ProgressColumn(
                "Risk %", min_value=0, max_value=100, format="%.0f%%"
            ),
            "risk_tier": st.column_config.TextColumn("Status"),
        },
        hide_index=True, height=280, use_container_width=True,
    )

    # ---------- Step 3: inspect one machine ----------
    st.divider()
    st.markdown('<p class="step-label">STEP 3</p>', unsafe_allow_html=True)
    st.subheader("Inspect a machine")

    options = fleet["unit"].tolist()
    selected = st.selectbox("Pick a machine to see the full picture", options)

    row = fleet[fleet["unit"] == selected].iloc[0]
    X_row = row[FEATURE_COLS].to_frame().T.astype(float)
    exp = explain_engine(model, explainer, X_row, row["unit"], row["pred_rul"])

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f"### Machine {exp['engine_id']}  {badge(exp['risk_tier'])}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Estimated **{exp['predicted_rul']} operating cycles** remaining "
            f"before failure risk becomes significant ({exp['risk_score_pct']}% risk score)."
        )
        st.markdown("**In plain English:**")
        use_live_api = st.toggle(
            "Use live Claude API", value=False,
            help="Requires ANTHROPIC_API_KEY. Off by default so browsing is free.",
        )
        if use_live_api and not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("No API key found - showing offline preview instead.")
            use_live_api = False
        with st.spinner("Thinking..."):
            try:
                recommendation = generate_recommendation(exp, mock=not use_live_api)
            except AdvisorError as e:
                st.error(str(e))
                recommendation = generate_recommendation(exp, mock=True)
                st.caption("Showing offline preview below instead:")
        st.info(recommendation)

    with right:
        st.caption("What's driving this reading:")
        st.pyplot(plot_factors(exp))


if __name__ == "__main__":
    main()
