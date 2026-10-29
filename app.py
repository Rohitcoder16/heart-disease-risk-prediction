"""Heart-disease risk triage demo  —  run with:  streamlit run app.py"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk Triage", page_icon="❤️", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load(os.path.join("model", "heart_model.joblib"))

@st.cache_data
def load_reference():
    return pd.read_csv("heart.csv") if os.path.exists("heart.csv") else None

art = load_artifact()
pipe, FEATURES = art["pipeline"], art["features"]
ref = load_reference()

st.title("❤️ Heart Disease Risk Triage")
st.caption("AIDS PBL project · decision-support prototype trained on 918 patients (Kaggle Heart Failure Prediction). "
           "**Not a medical device — does not replace clinical judgement.**")

# ------------------------------------------------------------------ inputs
with st.sidebar:
    st.header("Patient details")
    age = st.slider("Age (years)", 25, 80, 55)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    cp = st.selectbox("Chest-pain type", ["ASY", "NAP", "ATA", "TA"],
                      format_func=lambda x: {"ASY": "ASY – asymptomatic", "NAP": "NAP – non-anginal pain",
                                             "ATA": "ATA – atypical angina", "TA": "TA – typical angina"}[x])
    bp = st.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
    chol_missing = st.checkbox("Cholesterol not measured")
    chol = 0 if chol_missing else st.number_input("Serum cholesterol (mg/dl)", 80, 620, 220)
    fbs = st.radio("Fasting blood sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
    ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    maxhr = st.slider("Max heart rate achieved (bpm)", 60, 210, 140)
    ang = st.radio("Exercise-induced angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No", horizontal=True)
    oldpeak = st.slider("Oldpeak – ST depression (mm)", -3.0, 7.0, 0.5, 0.1)
    slope = st.selectbox("ST slope (peak exercise)", ["Up", "Flat", "Down"])
    st.divider()
    mode = st.radio("Screening mode", ["Standard", "High-sensitivity"],
                    help="Standard targets recall ≥ 0.90 (threshold %.2f). High-sensitivity targets recall ≥ 0.95 "
                         "(threshold %.2f) — more referrals, fewer missed patients; suited to screening camps."
                         % (art["threshold"], art["threshold_high_sensitivity"]))

patient = pd.DataFrame([{"Age": age, "Sex": sex, "ChestPainType": cp, "RestingBP": bp, "Cholesterol": chol,
                         "FastingBS": fbs, "RestingECG": ecg, "MaxHR": maxhr, "ExerciseAngina": ang,
                         "Oldpeak": oldpeak, "ST_Slope": slope}])[FEATURES]

# ------------------------------------------------------------------ prediction
p = float(pipe.predict_proba(patient)[0, 1])
thr = art["threshold"] if mode == "Standard" else art["threshold_high_sensitivity"]
low_max = art["tiers"]["low_max"]
if p < low_max:
    tier, colour, action = "LOW RISK", "🟢", "Routine follow-up, lifestyle advice, re-screen annually."
elif p < art["tiers"]["high_min"]:
    tier, colour, action = "MODERATE RISK", "🟠", "GP review within weeks; repeat tests (ECG / lipid profile); consider echo."
else:
    tier, colour, action = "HIGH RISK", "🔴", "Fast-track cardiology referral."
refer = p >= thr

c1, c2, c3 = st.columns([1.1, 1.1, 1.6])
c1.metric("Estimated risk score", f"{p:.0%}")
c1.progress(min(max(p, 0.0), 1.0))
c2.metric("Risk tier", f"{colour} {tier}")
c2.metric(f"Decision ({mode}, thr {thr:.2f})", "REFER" if refer else "No referral")
c3.subheader("Recommended action")
c3.info(action)
c3.caption("The score is a *risk rank* (tree-ensemble probabilities are slightly compressed), not an exact percentage.")

# ------------------------------------------------------------------ local explanation (occlusion)
st.subheader("Why this score? — what each input contributes")
if ref is not None:
    base = {c: (ref[c].median() if pd.api.types.is_numeric_dtype(ref[c]) and c != "FastingBS" else ref[c].mode()[0]) for c in FEATURES}
    base["Cholesterol"] = ref.loc[ref.Cholesterol > 0, "Cholesterol"].median()
    rows = []
    for f in FEATURES:
        alt = patient.copy(); alt[f] = base[f]
        rows.append((f, p - float(pipe.predict_proba(alt)[0, 1])))
    contrib = pd.DataFrame(rows, columns=["Feature", "Δ risk vs typical value"]).sort_values("Δ risk vs typical value")
    st.bar_chart(contrib.set_index("Feature"), horizontal=True, color="#e63946")
    st.caption("Occlusion analysis: change in predicted risk when a feature is replaced by the typical (median / most common) "
               "training value. Positive = pushes risk up.")
else:
    st.warning("heart.csv not found — explanation unavailable.")

with st.expander("Model details"):
    st.write(f"**Model:** {art['model_name']} · standard threshold {art['threshold']:.2f} · high-sensitivity threshold {art['threshold_high_sensitivity']:.2f}")
    st.write("Held-out test performance: recall 0.92, precision 0.85, F1 0.88, ROC-AUC 0.93. "
             "Trained on pooled 1980s–90s clinical data — validate locally before any real use.")
    st.dataframe(patient)
