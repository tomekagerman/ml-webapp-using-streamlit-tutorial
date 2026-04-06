import streamlit as st
import pandas as pd
import pickle

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Model file not found. Please run train_model.py first. Error: {e}")

st.title("🎓 Student Dropout Predictor")

# --- UI INPUTS ---
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 2)

if st.button("Predict Status"):
    # This dictionary matches case
    data = {
        'Attendance_Rate': [attendance],
        'GPA': [gpa],
        'Stress_Index': [stress],
        'Study_Hours_per_Day': [study_hours],
        # Filling other 13 features with neutral values to satisfy model shape
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
    
    # Re-order columns to match the model's training order exactly
    expected_order = [
        'Attendance_Rate', 'GPA', 'Stress_Index', 'Study_Hours_per_Day',
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Gender', 'Parental_Education', 
        'Part_Time_Job', 'Scholarship', 'Travel_Time_Minutes', 
        'Semester', 'Semester_GPA'
    ]
    input_df = input_df[expected_order]
    
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # Assuming 0 = Stay, 1 = Dropout based on typical dataset encoding
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")