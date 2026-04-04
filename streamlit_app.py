import streamlit as st
import pandas as pd
import pickle

# Load the new 4-feature model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# Input Fields
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.0)
attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 5.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)

if st.button("Predict Status"):
    # The order must match the 'features' list in train_model.py
    input_df = pd.DataFrame([[study_hours, stress, attendance, gpa]], 
                            columns=['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA'])
    
    prediction = model.predict(input_df)
    
    if prediction[0] == 1:
        st.error("### Result: High Risk of Dropout")
    else:
        st.success("### Result: Likely to Stay")