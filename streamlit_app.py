import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Student Dropout Predictor")

# 4 Inputs
input_data = np.array([
        22.4,   # Age
        1.0,    # Gender
        0.7,    # Ethnicity
        0.8,    # Parental Education
        2.5,    # Family Income
        gpa,    # Your GPA input
        attendance, # Your Attendance input
        0.6,    # Extracurriculars
        0.4,    # Part-time Job
        1.2,    # Self-study hours
        study_hours, # Your Study Hours input
        4.5,    # History of Failures
        0.8,    # Scholarship
        0.7,    # Internet Access
        stress, # Your Stress Index input
        2.1,    # Financial Stress
        0.6     # Peer Influence
    ])
    # Mapping to the correct columns (approximate based on standard dataset order)
    input_data[5] = gpa           # High importance
    input_data[6] = attendance    # High importance
    input_data[10] = study_hours
    input_data[14] = stress

if st.button("Predict Status"):
    # Create input dataframe
    input_data = pd.DataFrame([[study_hours, stress, attendance, gpa]], 
                            columns=['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA'])
    
    # Get prediction and the probability
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0] # Shows how confident the model is
    
    st.divider()

    # SWAPPED LOGIC: 
    if prediction == 0:
        st.success(f"### Result: Likely to Stay")
        st.info(f"Confidence: {proba[0]*100:.1f}%")
    else:
        st.error(f"### Result: High Risk of Dropout")
        st.info(f"Confidence: {proba[1]*100:.1f}%")