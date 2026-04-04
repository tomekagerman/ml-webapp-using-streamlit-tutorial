import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# 4 Inputs
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 2)

if st.button("Predict Status"):
    # Create input dataframe
    input_data = pd.DataFrame([[study_hours, stress, attendance, gpa]], 
                            columns=['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA'])
    
    # Get prediction and the probability
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0] # Shows how confident the model is
    
    st.divider()

    # SWAPPED LOGIC: 
    if prediction == 0:
        st.success(f"### Result: Likely to Stay")
        st.info(f"Confidence: {proba[0]*100:.1f}%")
    else:
        st.error(f"### Result: High Risk of Dropout")
        st.info(f"Confidence: {proba[1]*100:.1f}%")