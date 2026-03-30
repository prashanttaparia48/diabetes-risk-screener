import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Check",
    page_icon="🩺",
    layout="centered",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "model.pkl"),    "rb") as f: model    = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"),   "rb") as f: scaler   = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "features.pkl"), "rb") as f: features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, feature_cols = load_model()
except FileNotFoundError:
    st.error("Model not found. Please run `python src/train_model.py` first, then refresh this page.")
    st.stop()

st.markdown("""
    <h1 style='text-align:center; color:#C0392B;'>🩺 Diabetes Risk Check</h1>
    <p style='text-align:center; color:#666; font-size:15px;'>
        Answer a few quick health questions and get an instant risk estimate.<br>
        Takes about 30 seconds. No account needed.
    </p>
    <hr style='margin-bottom:24px;'>
""", unsafe_allow_html=True)

st.subheader("Tell us a bit about yourself")
st.caption("All values stay on your device — nothing is stored or sent anywhere.")

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("How many times have you been pregnant?", min_value=0,   max_value=20,  value=1,   step=1,   help="Enter 0 if you have never been pregnant or if you are male.")
    glucose        = st.number_input("Glucose level (mg/dL)",                  min_value=40,  max_value=300, value=110, step=1,   help="Plasma glucose from a blood test. Typical fasting range: 70–99 mg/dL.")
    blood_pressure = st.number_input("Blood pressure (mm Hg)",                 min_value=20,  max_value=180, value=72,  step=1,   help="Diastolic (lower) reading. Normal is around 60–80.")
    skin_thickness = st.number_input("Skin fold thickness (mm)",               min_value=0,   max_value=100, value=23,  step=1,   help="Triceps skinfold thickness. Used to estimate body fat.")

with col2:
    insulin = st.number_input("Insulin level (μU/mL)",           min_value=0,   max_value=900, value=79,  step=1,   help="2-hour serum insulin from an oral glucose tolerance test.")
    bmi     = st.number_input("BMI",                             min_value=10.0, max_value=70.0, value=25.0, step=0.1, format="%.1f", help="Body Mass Index = weight(kg) / height(m)². Normal: 18.5–24.9.")
    dpf     = st.number_input("Diabetes family history score",   min_value=0.0,  max_value=3.0,  value=0.47, step=0.01, format="%.2f", help="Diabetes Pedigree Function — estimates genetic risk based on family history. Higher = more family history.")
    age     = st.number_input("Your age",                        min_value=1,   max_value=120, value=30,  step=1)

st.markdown("<br>", unsafe_allow_html=True)
clicked = st.button("Check my risk →", use_container_width=True, type="primary")

if clicked:
    raw = {
        "Pregnancies":             pregnancies,
        "Glucose":                 glucose,
        "BloodPressure":           blood_pressure,
        "SkinThickness":           skin_thickness,
        "Insulin":                 insulin,
        "BMI":                     bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                     age,
        "BMI_Age":                 bmi * age,
        "Glucose_Insulin_Ratio":   glucose / (insulin + 1),
    }

    input_df = pd.DataFrame([raw])[feature_cols]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    if prob < 0.3:
        level, color, emoji, message = (
            "Low Risk", "#27AE60", "✅",
            "Your numbers look healthy. Keep up the good habits — regular exercise, balanced diet, and annual checkups go a long way."
        )
    elif prob < 0.6:
        level, color, emoji, message = (
            "Moderate Risk", "#E67E22", "⚠️",
            "A few of your readings are worth watching. Consider getting a proper blood test and speaking with a doctor — catching things early makes a big difference."
        )
    else:
        level, color, emoji, message = (
            "High Risk", "#C0392B", "🚨",
            "Some of your indicators are in a range associated with higher diabetes risk. Please see a doctor soon for a proper diagnosis. This is not a scare tactic — early detection genuinely saves lives."
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Your result")

    st.markdown(f"""
        <div style='background:{color}18; border-left:5px solid {color};
                    padding:18px 20px; border-radius:8px; margin-bottom:12px;'>
            <h2 style='color:{color}; margin:0 0 6px 0;'>{emoji} {level}</h2>
            <p style='font-size:15px; margin:0;'>{message}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**Estimated diabetes probability: {prob*100:.1f}%**")
    st.progress(float(prob))

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("What factors influenced this result?"):
        st.caption("These are the features the model weighted most heavily across all predictions — not just yours.")
        fi_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.bar_chart(fi_df.set_index("Feature")["Importance"])

    st.info(
        "**A word of honesty:** This tool was built as a student project using a "
        "publicly available dataset. It is not a medical device, and it has not been "
        "clinically validated. Think of it as a rough compass, not a diagnosis. "
        "If you're concerned about your health, please talk to a real doctor."
    )

st.markdown("""
    <hr>
    <p style='text-align:center; font-size:12px; color:#bbb;'>
        Built as a BYOP project · Scikit-learn + Streamlit · PIMA Diabetes Dataset
    </p>
""", unsafe_allow_html=True)
