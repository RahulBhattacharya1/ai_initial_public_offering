import os, json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IPO Outcome Predictor", page_icon="📈", layout="centered")
st.title("IPO Outcome Predictor")
st.write("Enter IPO attributes to estimate a **return/gain** or a **status** (auto-detected from your dataset).")

MODEL_PATH = os.path.join("models", "ipo_model.joblib")
SCHEMA_PATH = os.path.join("models", "schema.json")

def require(path, kind):
    if not os.path.exists(path):
        st.error(f"{kind} file missing: `{path}`. Upload it to your repo and restart.")
        st.stop()

require(MODEL_PATH, "Model")
require(SCHEMA_PATH, "Schema")

@st.cache_resource
def load_model_and_schema():
    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)
    model = joblib.load(MODEL_PATH)
    return model, schema

model, schema = load_model_and_schema()

task = schema.get("task", "regression")
target_name = schema.get("target", "listing_gain")
cat_schema = schema.get("categorical", {})
num_schema = schema.get("numeric_ranges", {})
feature_order = schema.get("feature_order", list(cat_schema.keys()) + list(num_schema.keys()))

with st.form("ipo_form"):
    inputs = {}
    # categorical controls
    for c, options in cat_schema.items():
        label = c.replace("_"," ").title()
        if options:
            inputs[c] = st.selectbox(label, options, index=0)
        else:
            inputs[c] = st.text_input(label)

    # numeric controls
    for c, rng in num_schema.items():
        mn, mx = float(rng.get("min", 0.0)), float(rng.get("max", 100.0))
        med = float(rng.get("median", (mn+mx)/2))
        if mn == mx:
            mx = mn + 1.0
        step = max(0.01, (mx - mn) / 100.0)
        inputs[c] = st.number_input(c.replace("_"," ").title(), min_value=mn, max_value=mx, value=med, step=step)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {k: inputs.get(k) for k in feature_order}
    X = pd.DataFrame([row])

    try:
        if task == "classification":
            pred = model.predict(X)[0]
            prob_txt = ""
            try:
                proba = model.predict_proba(X)[0]
                p = float(max(proba))
                prob_txt = f" (confidence: {p*100:.1f}%)"
            except Exception:
                pass
            st.success(f"Predicted {target_name.replace('_',' ').title()}: **{pred}**{prob_txt}")
        else:
            val = float(model.predict(X)[0])
            st.success(f"Predicted {target_name.replace('_',' ').title()}: **{val:,.3f}**")
        st.caption("Model: scikit-learn Pipeline (One-Hot for categoricals + Linear/Logistic Regression).")
    except Exception as e:
        st.error("Prediction failed. Ensure the model & schema match this app.")
        st.exception(e)

with st.expander("How it works"):
    st.markdown(
        """
        - **Auto-detected task**: classification if a status-like label exists; otherwise regression on a return/gain column.
        - **Preprocessing**: One-Hot for text features; numeric features passed through with median imputation at training time.
        - **Model size**: Compressed joblib to stay below GitHub’s 25 MB limit.
        """
    )
