import streamlit as st
import pandas as pd
import pickle

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(page_title="Student Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Predictor")

# --- ONLY 4 REQUESTED INDICATORS ---
gpa = st.number_input("Cumulative GPA", min_value=0.0, max_value=4.0, value=2.5, step=0.01)
attendance = st.slider("Attendance Rate (%)", 0, 100, 70)
study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=10)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)

if st.button("Predict Status"):
    # This dictionary matches the EXACT feature names and order the model expects
    # based on the error: GPA, Gender, Parental_Education, Part_Time_Job, Scholarship, etc.
    data = {
        'GPA': [gpa],
        'Attendance_Rate': [attendance],
        'Study_Hours_per_Day': [study_hours / 7], # Converting weekly to daily if needed
        'Stress_Index': [stress],
        'Age': [20],
        'Gender': [0],
        'Family_Income': [20000],
        'Internet_Access': [1],
        'Assignment_Delay_Days': [2],
        'Travel_Time_Minutes': [30],
        'Part_Time_Job': [0],
        'Scholarship': [0],
        'Semester_GPA': [gpa],
        'CGPA': [gpa],
        'Semester': [2],
        'Department': [1],
        'Parental_Education': [1]
    }
    
    input_data = pd.DataFrame(data)
    
    # Reorder columns to match exactly the training model
    prediction = model.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
        st.write("Current indicators suggest the student is at risk.")
    else:
        st.success("### Result: Likely to Stay")
        st.write("Student is currently on a stable path.")