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
    # ORDER MATTERS: Study_Hours, Stress, Attendance, GPA
    input_data = pd.DataFrame([[study_hours, stress, attendance, gpa]], 
                            columns=['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA'])
    
    prediction = model.predict(input_data)[0]
    
    st.divider()
    # In most ML datasets: 1 = Stay/Success, 0 = Dropout/Failure
    if prediction == 1:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")