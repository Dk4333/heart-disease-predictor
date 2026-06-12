"""
app.py — Heart Disease Risk Predictor
Streamlit application with live XGBoost inference + SHAP waterfall explanation.

Run: streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.features import engineer_features, ALL_FEATURES

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

st.title("🫀 Heart Disease Risk Predictor")
st.caption(
    "XGBoost + SHAP explainability · UCI Heart Disease Dataset · "
    "Threshold tuned for ≥ 90% recall"
)
st.divider()


# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artefacts():
    import subprocess
    try:
        pipeline = joblib.load("models/heart_disease_pipeline.pkl")
        explainer = joblib.load("models/shap_explainer.pkl")
        with open("models/threshold.txt") as f:
            threshold = float(f.read().strip())
        return pipeline, explainer, threshold
    except Exception as e:
        st.warning("⚠️ Model compatibility issue detected. Retraining the model in the cloud environment... (This will take about 20 seconds)")
        subprocess.run([sys.executable, "src/train.py"], check=True)
        # Try loading again after retraining
        pipeline = joblib.load("models/heart_disease_pipeline.pkl")
        explainer = joblib.load("models/shap_explainer.pkl")
        with open("models/threshold.txt") as f:
            threshold = float(f.read().strip())
        st.success("Retraining complete!")
        return pipeline, explainer, threshold


try:
    pipeline, explainer, threshold = load_artefacts()
except Exception as e:
    st.error(f"Critical error loading or training the model: {e}")
    st.stop()


# ── Sidebar — patient inputs ──────────────────────────────────────────────────

st.sidebar.header("Patient clinical values")

age       = st.sidebar.slider("Age", 20, 80, 55)
sex       = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
cp        = st.sidebar.selectbox(
    "Chest pain type",
    [0, 1, 2, 3],
    format_func=lambda x: {
        0: "0 — Asymptomatic (high risk)",
        1: "1 — Atypical angina",
        2: "2 — Non-anginal pain",
        3: "3 — Typical angina",
    }[x],
)
trestbps  = st.sidebar.slider("Resting BP (mmHg)", 90, 200, 130)
chol      = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 250)
fbs       = st.sidebar.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
restecg   = st.sidebar.selectbox(
    "Resting ECG result",
    [0, 1, 2],
    format_func=lambda x: {0: "0 — Normal", 1: "1 — ST-T abnormality", 2: "2 — LV hypertrophy"}[x],
)
thalach   = st.sidebar.slider("Max heart rate achieved", 70, 210, 150)
exang     = st.sidebar.selectbox("Exercise-induced angina", [0, 1],
                                  format_func=lambda x: "Yes" if x else "No")
oldpeak   = st.sidebar.slider("ST depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
slope     = st.sidebar.selectbox(
    "Slope of peak ST segment",
    [0, 1, 2],
    format_func=lambda x: {0: "0 — Upsloping", 1: "1 — Flat", 2: "2 — Downsloping"}[x],
)
ca        = st.sidebar.selectbox("Major vessels coloured (ca)", [0, 1, 2, 3])
thal      = st.sidebar.selectbox(
    "Thalassemia",
    [0, 1, 2, 3],
    format_func=lambda x: {0: "0 — Normal", 1: "1 — Fixed defect",
                            2: "2 — Reversible defect", 3: "3 — Unknown"}[x],
)


# ── Build input dataframe ─────────────────────────────────────────────────────

raw_input = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
}])

input_df = engineer_features(raw_input)[ALL_FEATURES]


# ── Predict & display ─────────────────────────────────────────────────────────

proba     = pipeline.predict_proba(input_df)[0][1]
is_high   = proba >= threshold
risk_text = "High risk" if is_high else "Low risk"
risk_color = "#c0392b" if is_high else "#1D9E75"

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric("Disease probability", f"{proba:.1%}")
    st.progress(float(proba))

with col2:
    st.markdown(
        f"<div style='padding:16px 20px; border-radius:10px; border:1.5px solid {risk_color};"
        f"text-align:center'>"
        f"<p style='margin:0;font-size:13px;color:gray'>Prediction</p>"
        f"<p style='margin:4px 0 0;font-size:22px;font-weight:600;color:{risk_color}'>{risk_text}</p>"
        f"<p style='margin:4px 0 0;font-size:12px;color:gray'>threshold = {threshold:.2f}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"**Interpretation:** At the tuned threshold of `{threshold:.2f}`, "
        f"this model achieves ≥ 90% recall on the test set — meaning it misses fewer than 10% "
        f"of true disease cases. The probability shown is the raw XGBoost output before thresholding."
    )

st.divider()


# ── SHAP waterfall for this patient ──────────────────────────────────────────

st.subheader("Why this prediction?")
st.caption("SHAP waterfall — contribution of each feature to this patient's risk score")

X_proc = pipeline.named_steps["preprocessor"].transform(input_df)
sv     = explainer.shap_values(X_proc)

fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.waterfall(
    shap.Explanation(
        values=sv[0],
        base_values=explainer.expected_value,
        data=X_proc[0],
        feature_names=ALL_FEATURES,
    ),
    show=False,
    max_display=12,
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.caption(
    "Red bars push the prediction toward disease (higher risk). "
    "Blue bars push toward no disease (lower risk). "
    "The base value is the model's average prediction across the training set."
)

st.divider()


# ── Global feature importance ─────────────────────────────────────────────────

with st.expander("Global feature importance (test set)"):
    if os.path.exists("outputs/shap_bar.png"):
        st.image("outputs/shap_bar.png", caption="Mean |SHAP value| across test set")
    else:
        st.info("Run `python src/train.py` to generate SHAP summary plots.")

with st.expander("Model performance"):
    st.markdown("""
| Metric | Score |
|---|---|
| ROC-AUC | ~0.91 |
| Accuracy | ~0.87 |
| Recall (disease) | ≥ 0.90 (tuned) |
| Precision (disease) | ~0.88 |
| F1 Score | ~0.90 |

Dataset: UCI Heart Disease · 303 patients · 5-fold stratified CV during tuning.
    """)
    if os.path.exists("outputs/roc_curve.png"):
        c1, c2 = st.columns(2)
        c1.image("outputs/roc_curve.png")
        if os.path.exists("outputs/confusion_matrix.png"):
            c2.image("outputs/confusion_matrix.png")
