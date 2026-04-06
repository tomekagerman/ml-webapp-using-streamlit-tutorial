import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎓 Student Dropout Predictor")

# 2. Create Inputs
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.0)
attendance = st.slider("Attendance %", 0, 100, 85)
study_hours = st.number_input("Weekly Study Hours", 0, 100, 10)
stress = st.slider("Stress Index (1-10)", 1, 10, 5)

# 3. Prediction Logic with Averages
if st.button("Predict Dropout Status"):
    input_data = np.array([
        22.4, 1.0, 0.7, 0.8, 2.5,  # Hidden averages
        gpa, attendance,           # User inputs
        0.6, 0.4, 1.2, study_hours, 
        4.5, 0.8, 0.7, stress, 
        2.1, 0.6
    ])
    
    prediction = model.predict([input_data])
    
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
    else:
        st.success("### Result: Likely to Stay")