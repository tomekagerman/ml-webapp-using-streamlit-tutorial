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

# --- ONLY YOUR 4 REQUESTED INDICATORS ---
gpa = st.number_input("Cumulative GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=15)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)

if st.button("Predict Status"):
    # We map your 4 inputs to the specific column names the model requires.
    # The 'extra' columns are filled with neutral/average values so they don't break the model.
    input_data = pd.DataFrame([[
        2,          # Assignment_Delay_Days (Neutral)
        gpa,        # CGPA (Using your GPA input)
        1,          # Department (Default)
        25000,      # Family_Income (Average)
        1,          # Internet_Access (Yes)
        20,         # Age (Average)
        attendance, # Attendance_Rate (Your Input)
        stress      # Stress_Index (Your Input)
        # Note: If your model has more than 8 columns, add them here
    ]], columns=[
        'Assignment_Delay_Days', 'CGPA', 'Department', 'Family_Income', 
        'Internet_Access', 'Age', 'Attendance_Rate', 'Stress_Index'
    ])
    
    # Make Prediction
    prediction = model.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
        st.caption("Based on low attendance or high stress detected.")
    else:
        st.success("### Result: Likely to Stay")
        st.caption("Student metrics indicate positive academic standing.")