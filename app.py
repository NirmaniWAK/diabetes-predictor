import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

# Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health parameters to check if you're likely to have diabetes.")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Predict", "About Dataset"])

if page == "Predict":
    st.header("Enter Patient Details")

    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    Glucose = st.slider("Glucose", 0, 200, 120)
    BloodPressure = st.slider("Blood Pressure", 0, 122, 70)
    SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.slider("Insulin", 0, 846, 79)
    BMI = st.slider("BMI", 0.0, 70.0, 32.0)
    DPF = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.slider("Age", 10, 100, 33)

    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    input_scaled = scaler.transform(input_data)

    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        st.success(f"The patient is likely: {result}")

elif page == "About Dataset":
    st.header("📊 Dataset Information")
    st.write("""
    The Pima Indians Diabetes dataset contains health information and a binary outcome (diabetic or not).
    Features include:
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age
    """)