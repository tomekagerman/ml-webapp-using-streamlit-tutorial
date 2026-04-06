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
    # Create a dictionary with ALL columns the model expects (17 total)
    # Use inputs for the first 4, and 'safe' averages for the rest
    data = {
        'Study_Hours_per_Day': [study_hours],
        'Stress_Index': [stress],
        'Attendance_Rate': [attendance],
        'GPA': [gpa],
        'Age': [20],                   # Average age
        'Gender': [0],                 # Neutral
        'Family_Income': [25000],      # Average income
        'Internet_Access': [1],        # Assume they have it
        'Assignment_Delay_Days': [1],  # Low delay
        'Travel_Time_Minutes': [30],   # Average commute
        'Part_Time_Job': [0],          # No job
        'Scholarship': [0],            # No scholarship
        'Semester_GPA': [gpa],         # Match their current GPA
        'CGPA': [gpa],                 # Match their current GPA
        'Semester': [2],               # Mid-program
        'Department': [1],             # General dept
        'Parental_Education': [2]      # Average level
    }
    
    input_df = pd.DataFrame(data)
    
    # Get the prediction
    prediction = model.predict(input_df)[0]
    
    st.divider()
    # Logic: 0 is Stay, 1 is Dropout
    if prediction == 0:
        st.success("### Result: Likely to Stay")
    else:
        st.error("### Result: High Risk of Dropout")