"""Streamlit demo: score one transaction and explain the decision.

Run with:  streamlit run app.py
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.explain import FEATURE_LABELS, explain_transaction, load_headline_model, shap_explanation
from src.features import build_features

st.set_page_config(page_title="Fraud check demo", page_icon="🔎", layout="centered")

PRESETS = {
    "legit": {"type": "CASH_OUT", "amount": 12_500.0, "orig_before": 48_000.0,
              "orig_after": 35_500.0, "dest_before": 150_000.0, "dest_after": 162_500.0, "hour": 14},
    "takeover": {"type": "TRANSFER", "amount": 181_000.0, "orig_before": 181_000.0,
                 "orig_after": 0.0, "dest_before": 0.0, "dest_after": 181_000.0, "hour": 3},
}


def apply_preset(name: str) -> None:
    """Copy a preset into the input widgets' session state."""
    st.session_state.update(PRESETS[name])


@st.cache_resource
def get_model():
    """Load the saved XGBoost model and threshold once per server process."""
    return load_headline_model()


if "amount" not in st.session_state:
    apply_preset("legit")

model, threshold = get_model()

# ---------------------------------------------------------------- sidebar inputs
with st.sidebar:
    st.header("Transaction")
    col1, col2 = st.columns(2)
    col1.button("Typical legit transaction", key="preset_legit", on_click=apply_preset,
                args=("legit",), width="stretch")
    col2.button("Account-takeover pattern", key="preset_takeover", on_click=apply_preset,
                args=("takeover",), width="stretch")

    st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"], key="type")
    st.number_input("Amount", min_value=0.0, step=1_000.0, format="%.2f", key="amount")
    st.number_input("Origin balance before", min_value=0.0, step=1_000.0, format="%.2f",
                    key="orig_before")
    st.number_input("Origin balance after", min_value=0.0, step=1_000.0, format="%.2f",
                    key="orig_after")
    st.number_input("Destination balance before", min_value=0.0, step=1_000.0, format="%.2f",
                    key="dest_before")
    st.number_input("Destination balance after", min_value=0.0, step=1_000.0, format="%.2f",
                    key="dest_after")
    st.slider("Hour of day", 0, 23, key="hour")
    st.caption("Balances *after* the transaction are shown for completeness but are not "
               "used by the model: a bank would not know them when deciding.")

# ---------------------------------------------------------------- scoring
ss = st.session_state
row = {
    "step": 24 + ss.hour,              # build_features derives hour as step % 24
    "type": ss.type,
    "amount": ss.amount,
    "oldbalanceOrg": ss.orig_before,
    "newbalanceOrig": ss.orig_after,
    "oldbalanceDest": ss.dest_before,
    "newbalanceDest": ss.dest_after,
}

st.title("Transaction fraud check")
st.caption("Demo only. The model was trained on PaySim, a **synthetic** mobile-money "
           "dataset, and must not be used for real decisions.")

if ss.amount <= 0:
    st.warning("Enter an amount above 0 to score the transaction.")
    st.stop()

result = explain_transaction(row, model=model, threshold=threshold)

col_score, col_decision = st.columns([1, 2])
col_score.metric("Fraud probability", f"{result['score']:.1%}")
with col_decision:
    if result["flagged"]:
        st.error(f"**FLAG**: send for analyst review (score {result['score']:.3f} "
                 f">= threshold {threshold:.3f})")
    else:
        st.success(f"**ALLOW**: no review needed (score {result['score']:.3f} "
                   f"< threshold {threshold:.3f})")
st.caption("The threshold minimises expected cost on validation data, assuming a missed "
           "fraud costs 500 and a false alarm costs 5.")

st.subheader("Top reasons")
for feature in result["top_features"]:
    arrow = "▲" if feature["shap"] > 0 else "▼"
    st.markdown(f"{arrow} {feature['text']}")

# ---------------------------------------------------------------- details
X = build_features(pd.DataFrame([row]), feature_set="pre_transaction")

with st.expander("SHAP waterfall (how each feature moved the score)"):
    plt.figure()
    shap.plots.waterfall(shap_explanation(model, X)[0], max_display=8, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Values are in log-odds. Red bars push towards fraud, blue bars away from it.")

with st.expander("Model inputs (engineered features)"):
    st.dataframe(X.rename(columns=FEATURE_LABELS).T.rename(columns={0: "value"}),
                 width="stretch")
