import streamlit as st
import requests
import joblib
import numpy as np
st.set_page_config(page_title="Placement Predictor", layout="centered")

def load_assets():
    model = joblib.load('placement_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()
st.title("Student Placement Prediction")

st.write("Enter your details to check if you are eligible for placement.")


cgpa=st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.01)
backlogs=st.number_input("Number of Backlogs:", min_value=0, step=1)

if st.button("Check Eligibility"):
    input_data = np.array([[cgpa, backlogs]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Eligible for Placement")
    else:
        st.error("❌ Not Eligible for Placement")
