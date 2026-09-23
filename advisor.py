"""
PredictIQ - Day 4: Claude API advisory layer.
Takes the structured SHAP explanation dict from shap_explain.py and translates it
into a plain-English, actionable recommendation for a maintenance technician.

The LLM does NOT do any prediction or analysis of its own -- it only formats
numbers we already computed (RUL, risk %, SHAP factors) into readable language.
This keeps the model swap-able and the actual engineering (XGBoost + SHAP)
auditable independent of the LLM.

Requires: pip install anthropic
Requires: environment variable ANTHROPIC_API_KEY set to a valid key.

Usage:
    python3 advisor.py            # live mode, calls the real API
    python3 advisor.py --mock     # offline mode, no API key/cost, for testing
      the formatting logic and demoing the pipeline without burning API credits
"""
import os
import sys
import json


def format_prompt(explanation: dict) -> str:
    """Turn a shap_explain.py explanation dict into the prompt Claude sees."""
    risk_lines = "\n".join(
        f"  - {f['feature_name']}: reading={f['raw_reading']} "
        f"(SHAP {f['shap_value']}, {f['pct_contribution']}% of risk pull)"
        for f in explanation["top_risk_factors"]
    )
    protective_lines = "\n".join(
        f"  - {f['feature_name']}: reading={f['raw_reading']} "
        f"(SHAP {f['shap_value']}, {f['pct_contribution']}% of protective pull)"
        for f in explanation["top_protective_factors"]
    )
    ratio = (
        abs(explanation["net_negative_contribution"])
        / max(abs(explanation["net_positive_contribution"]), 0.01)
    )

    return f"""You are a maintenance advisor for a turbofan engine fleet. Given this \
equipment risk analysis, write a 2-3 sentence plain-English explanation and a \
specific, actionable recommendation for a maintenance technician. Be concrete about \
urgency (do not use vague terms like "soon" -- reference the actual cycle estimate \
given). Do not invent numbers beyond what is provided below.

Engine ID: {explanation['engine_id']}
Predicted Remaining Useful Life: {explanation['predicted_rul']} cycles
Risk Score: {explanation['risk_score_pct']}% ({explanation['risk_tier']})
Risk factors outweigh protective factors by roughly {ratio:.1f}x

Top risk-increasing readings:
{risk_lines}

Top protective (reassuring) readings:
{protective_lines}
"""


def call_claude(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        raise AdvisorError(
            "Your Claude API key looks invalid or expired. Double-check "
            "ANTHROPIC_API_KEY and try again."
        )
    except anthropic.PermissionDeniedError:
        raise AdvisorError(
            "Your Claude API key doesn't have permission for this request. "
            "Check your account's plan/access in the Anthropic Console."
        )
    except anthropic.RateLimitError:
        raise AdvisorError(
            "Hit the API's rate limit. Wait a few seconds and try again."
        )
    except anthropic.BadRequestError as e:
        msg = str(e)
        if "credit balance" in msg.lower():
            raise AdvisorError(
                "Your Anthropic account is out of API credits. Add credits in "
                "Plans & Billing, or switch this toggle off to use offline preview "
                "mode instead."
            )
        raise AdvisorError(f"The API rejected this request: {msg}")
    except anthropic.APIConnectionError:
        raise AdvisorError(
            "Couldn't reach the Claude API — check your internet connection."
        )
    except anthropic.APIStatusError as e:
        raise AdvisorError(f"Claude API error (status {e.status_code}): {e.message}")


class AdvisorError(Exception):
    """Raised when the live Claude API call fails, with a message safe to show
    directly in the UI (no stack trace, no internal details)."""
    pass


def mock_claude(prompt: str, explanation: dict) -> str:
    """Deterministic stand-in so the pipeline can be demoed/tested without an
    API key or cost. Mirrors the shape of a real response."""
    tier = explanation["risk_tier"]
    rul = explanation["predicted_rul"]
    top = explanation["top_risk_factors"][0]["feature_name"]
    if tier == "HIGH":
        return (
            f"[MOCK] Engine {explanation['engine_id']} shows a {tier.lower()} failure "
            f"risk with an estimated {rul} cycles of life remaining, driven primarily "
            f"by an abnormal {top.lower()} reading. Recommend scheduling an inspection "
            f"immediately, before the estimated {rul}-cycle window closes, to avoid "
            f"unplanned downtime."
        )
    else:
        return (
            f"[MOCK] Engine {explanation['engine_id']} is operating within normal "
            f"parameters with an estimated {rul} cycles of remaining life. No "
            f"immediate action needed; continue routine monitoring."
        )


def generate_recommendation(explanation: dict, mock=False) -> str:
    prompt = format_prompt(explanation)
    if mock:
        return mock_claude(prompt, explanation)
    return call_claude(prompt)


if __name__ == "__main__":
    from baseline_model import load, split_by_engine, FEATURE_COLS
    from shap_explain import train_xgb, explain_engine
    import shap

    mock = "--mock" in sys.argv

    train, test = load()
    tr, val = split_by_engine(train)
    X_tr, y_tr = tr[FEATURE_COLS], tr["RUL"]
    model = train_xgb(X_tr, y_tr)
    explainer = shap.TreeExplainer(model)

    test_last = test.loc[test.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
    preds = model.predict(test_last[FEATURE_COLS])
    test_last["pred_rul"] = preds

    highest_risk = test_last.nsmallest(2, "pred_rul")
    lowest_risk = test_last.nlargest(1, "pred_rul")
    import pandas as pd
    sample = pd.concat([highest_risk, lowest_risk])

    if not mock and not os.environ.get("ANTHROPIC_API_KEY"):
        print("No ANTHROPIC_API_KEY found in environment. Run with --mock to test "
              "the formatting logic offline, or set the key and rerun for live output.")
        sys.exit(1)

    for _, row in sample.iterrows():
        X_row = row[FEATURE_COLS].to_frame().T
        exp = explain_engine(model, explainer, X_row, row["unit"], row["pred_rul"])
        recommendation = generate_recommendation(exp, mock=mock)
        print(f"--- Engine {exp['engine_id']} ({exp['risk_tier']}, "
              f"RUL={exp['predicted_rul']}) ---")
        print(recommendation)
        print()
