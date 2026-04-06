import streamlit as st
import pandas as pd
import pickle

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Model file (model.pkl) not found. Please run 'python3 train_model.py' in the terminal.")

st.title("🎓 Student Dropout Predictor")

# UI Inputs for the 4 features you want to control
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 2)

if st.button("Predict Status"):
    # This dictionary uses the EXACT capitalized names the model expects
    data = {
        'Attendance_Rate': [attendance],
        'GPA': [gpa],
        'Stress_Index': [stress],
        'Study_Hours_per_Day': [study_hours],
        'Assignment_Delay_Days': [1],
        'CGPA': [gpa],
        'Department': [1],
        'Family_Income': [25000],
        'Internet_Access': [1],
        'Age': [20],
        'Gender': [0],
        'Parental_Education': [2],
        'Part_Time_Job': [0],
        'Scholarship': [0],
        'Travel_Time_Minutes': [30],
        'Semester': [2],
        'Semester_GPA': [gpa]
    }
    
    input_df = pd.DataFrame(data)
    
    # CRITICAL: The features MUST be in this exact order to match the model
    expected_order = [
        'Attendance_Rate', 'GPA', 'Stress_Index', 'Study_Hours_per_Day',
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Gender', 'Parental_Education', 
        'Part_Time_Job', 'Scholarship', 'Travel_Time_Minutes', 
        'Semester', 'Semester_GPA'
    ]
    
    input_df = input_df[expected_order]
    
    # Perform prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    
    # Standard Encoding: 0 is Stay, 1 is Dropout
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")