import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# 4 UI Inputs
gpa_input = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance_input = st.slider("Attendance Rate (%)", 0, 100, 95)
study_input = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress_input = st.slider("Stress Level (1-10)", 1, 10, 2)

if st.button("Predict Status"):
    # These keys MUST match the "Seen at fit time" list in your error exactly
    data = {
        'Attendance_Rate': [attendance_input],
        'GPA': [gpa_input],
        'Stress_Index': [stress_input],
        'Study_Hours_per_Day': [study_input],
        'Assignment_Delay_Days': [1],
        'CGPA': [gpa_input],
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
        'Semester_GPA': [gpa_input]
    }
    
    input_df = pd.DataFrame(data)
    
    # We must also ensure the columns are in the order the model expects.
    # Based on your error, these 4 were listed as missing from your previous lowercase attempt:
    expected_order = [
        'Attendance_Rate', 'GPA', 'Stress_Index', 'Study_Hours_per_Day',
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Gender', 'Parental_Education', 
        'Part_Time_Job', 'Scholarship', 'Travel_Time_Minutes', 
        'Semester', 'Semester_GPA'
    ]
    
    input_df = input_df[expected_order]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # If 1.0 GPA/10% Attendance shows "Stay", change this to 'if prediction == 1'
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")