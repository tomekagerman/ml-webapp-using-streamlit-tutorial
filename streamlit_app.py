import streamlit as st
import pandas as pd
import pickle

# 1. Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# 2. Setup the 4 Inputs
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 2)

# 3. Prediction Logic
if st.button("Predict Status"):
    # This dictionary contains EVERY feature the model expects 
    # We use 4 UI inputs and fill the rest with 'neutral' average values
    data = {
        'Assignment_Delay_Days': [1],
        'CGPA': [gpa],
        'Department': [1],
        'Family_Income': [25000],
        'Internet_Access': [1],
        'Age': [20],
        'Attendance_Rate': [attendance],
        'Stress_Index': [stress],
        'Study_Hours_per_Day': [study_hours],
        'Gender': [0],
        'Parental_Education': [2],
        'Part_Time_Job': [0],
        'Scholarship': [0],
        'Travel_Time_Minutes': [30],
        'GPA': [gpa],
        'Semester': [2],
        'Semester_GPA': [gpa]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(data)
    
    # The model expects a very specific column order. 
    # This line ensures the columns are arranged exactly as they were during training.
    expected_columns = [
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Attendance_Rate', 'Stress_Index',
        'Study_Hours_per_Day', 'Gender', 'Parental_Education', 'Part_Time_Job',
        'Scholarship', 'Travel_Time_Minutes', 'GPA', 'Semester', 'Semester_GPA'
    ]
    input_df = input_df[expected_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # 0 = Stay, 1 = Dropout (Standard encoding for this dataset)
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")