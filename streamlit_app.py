import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained model
@st.cache_resource # This keeps the model in memory for faster performance
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# 2. Set up the Page UI
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

st.title("🎓 Student Success Predictor")
st.markdown("""
This app predicts whether a student is likely to **Stay** or **Dropout** based on academic and personal factors.
---
""")

# 3. Create Input Fields (Replacing the HTML Form)
col1, col2 = st.columns(2)

with col1:
    GPA = st.number_input("Current GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    Attendance_Rate = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)

with col2:
    Study_Hours_per_Day = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=15)
    Stress_Index = st.slider("Stress Level (1-10)", 1, 10, 5)

# 4. Logic for Prediction
if st.button("Predict Result"):
    # The error says your model wants exactly 4 features.
    # We must pass them in the SAME order they were trained (GPA, Attendance, Study Hours, Stress).
    input_features = np.array([[Study_Hours_per_Day, Stress_Index, Attendance_Rate, GPA]])

    # Make the prediction
    prediction = model.predict(input_features)
    
    # Display Result
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
    else:
        st.success("### Result: Likely to Stay")

st.info("Note: This model uses neutral averages for features not collected in this form.")