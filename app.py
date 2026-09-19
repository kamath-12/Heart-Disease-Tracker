import streamlit as st
import joblib
import numpy as np
import warnings, sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model and scaler once at the start
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction App")
st.write("Enter patient details below to predict heart disease:")

# Input fields for all 13 features
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ("Male", "Female"))
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (0=No, 1=Yes)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of peak exercise ST segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", [1, 2, 3])

# Convert categorical fields to numeric
sex = 1 if sex == "Male" else 0

# Predict button
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")
