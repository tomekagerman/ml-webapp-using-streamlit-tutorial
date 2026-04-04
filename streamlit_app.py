import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎓 Student Dropout Predictor")
st.write("Enter student metrics below to predict enrollment status.")

# 2. User Input Widgets
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.9)
attendance = st.slider("Attendance Percentage", 0, 100, 95)
study_hours = st.number_input("Weekly Study Hours", 0, 100, 15)
stress = st.slider("Stress Index (1-10)", 1, 10, 2)

# 3. Prediction Logic (Triggered by Button)
if st.button("Predict Dropout Status"):
    # We use 'Safe' background values to allow a "Stay" prediction
    input_data = np.array([
        22.4, 1.0, 0.7, 0.8, 3.5,  # Age, Gender, Eth, Edu, Income
        gpa,                       # User Input
        attendance,                # User Input
        0.6, 0.4, 1.2,             # Extra, Job, Self-study
        study_hours,               # User Input
        0.0,                       # <--- FIX: History of Failures set to 0
        0.8, 0.7,                  # Scholarship, Internet
        stress,                    # User Input
        1.1, 0.6                   # Lowered Financial Stress
    ])
    
    # Predict
    prediction = model.predict([input_data])
    
    # 4. Display Results
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
        st.warning("Recommendation: Early intervention advised.")
    else:
        st.success("### Result: Likely to Stay")
        st.balloons()