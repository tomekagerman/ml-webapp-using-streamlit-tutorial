import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 2.0) # Lowered default to test
attendance = st.slider("Attendance Rate (%)", 0, 100, 50) # Lowered default
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 2.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 8)

if st.button("Predict Status"):
    input_df = pd.DataFrame([[study_hours, stress, attendance, gpa]], 
                            columns=['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA'])
    
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # Logic check: Most datasets use 0 for Dropout and 1 for Stay
    if prediction == 0: 
        st.error("### Result: High Risk of Dropout")
    else:
        st.success("### Result: Likely to Stay")