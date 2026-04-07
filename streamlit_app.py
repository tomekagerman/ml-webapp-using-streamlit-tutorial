import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Model file not found. Run 'python3 train_model.py' first.")

st.title("🎓 Student Dropout Predictor")

# UI Inputs (The 4 to control)
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.0)
attendance = st.slider("Attendance %", 0, 100, 85)
study_hours = st.number_input("Weekly Study Hours", 0.0, 168.0, 10.0)
stress = st.slider("Stress Index (1-10)", 1, 10, 5)

if st.button("Predict Dropout Status"):
    # We must provide ALL 17 features that the model was trained on
    # Capitalization must match the error log exactly (Age, Attendance_Rate, etc.)
    data = {
        'Attendance_Rate': [attendance],
        'GPA': [gpa],
        'Stress_Index': [stress],
        'Study_Hours_per_Day': [study_hours / 7], # Converting weekly to daily
        'Assignment_Delay_Days': [0],
        'CGPA': [gpa],
        'Department': [1],
        'Family_Income': [30000],
        'Internet_Access': [1],
        'Age': [20],
        'Gender': [0],
        'Parental_Education': [2],
        'Part_Time_Job': [0],
        'Scholarship': [0],
        'Travel_Time_Minutes': [30],
        'Semester': [1],
        'Semester_GPA': [gpa]
    }
    
    input_df = pd.DataFrame(data)
    
    # Order the columns exactly how the model expects them
    expected_columns = [
        'Attendance_Rate', 'GPA', 'Stress_Index', 'Study_Hours_per_Day',
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Gender', 'Parental_Education', 
        'Part_Time_Job', 'Scholarship', 'Travel_Time_Minutes', 
        'Semester', 'Semester_GPA'
    ]
    input_df = input_df[expected_columns]
    
    # Run prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # 0 = Likely to Stay, 1 = High Risk (Standard dataset encoding)
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")