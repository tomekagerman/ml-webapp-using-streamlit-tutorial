import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# 4 UI Inputs
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.5)
attendance = st.slider("Attendance Rate (%)", 0, 100, 95)
study_hours = st.number_input("Daily Study Hours", 0.0, 24.0, 6.0)
stress = st.slider("Stress Level (1-10)", 1, 10, 2)

if st.button("Predict Status"):
    # All keys are now LOWERCASE to match the model's training requirements
    data = {
        'assignment_delay_days': [1],
        'cgpa': [gpa],
        'department': [1],
        'family_income': [25000],
        'internet_access': [1],
        'age': [20],
        'attendance_rate': [attendance],
        'stress_index': [stress],
        'study_hours_per_day': [study_hours],
        'gender': [0],
        'parental_education': [2],
        'part_time_job': [0],
        'scholarship': [0],
        'travel_time_minutes': [30],
        'gpa': [gpa],
        'semester': [2],
        'semester_gpa': [gpa]
    }
    
    input_df = pd.DataFrame(data)
    
    # Ensuring the order is also lowercase
    expected_columns = [
        'assignment_delay_days', 'cgpa', 'department', 'family_income', 
        'internet_access', 'age', 'attendance_rate', 'stress_index',
        'study_hours_per_day', 'gender', 'parental_education', 'part_time_job',
        'scholarship', 'travel_time_minutes', 'gpa', 'semester', 'semester_gpa'
    ]
    input_df = input_df[expected_columns]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # 0 = Stay, 1 = Dropout
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")