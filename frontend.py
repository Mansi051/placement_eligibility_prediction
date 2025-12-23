import streamlit as st
import requests

st.title("Student Placement Prediction")

st.write("Enter your details to check if you are eligible for placement.")


cgpa=st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, step=0.01)
backlogs=st.number_input("Number of Backlogs:", min_value=0, step=1)

if st.button("Check Eligibility"):
    api_url = "https://Mansi051/student-placement-api/predict"
    payload = {"cgpa": cgpa, "backlogs": backlogs}
    
    try:
        response = requests.post(api_url, json=payload)
        result = response.json()
        st.success(f"Placement Status: {result['placement_status']}")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
