import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the trained model
@st.cache_resource # This keeps the model in memory for faster performance
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

model = load_model()

# 2. Set up the Page UI
st.set_page_config(page_title="Student Dropout Predictor", layout="centered")

st.title("🎓 Student Success Predictor")
st.markdown("""
This app predicts whether a student is likely to **Stay** or **Dropout** based on academic and personal factors.
---
""")

# 3. Create Input Fields (Replacing the HTML Form)
col1, col2 = st.columns(2)

with col1:
    gpa = st.number_input("Current GPA", min_value=0.0, max_value=4.0, value=3.0, step=0.1)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)

with col2:
    study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=15)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

# 4. Logic for Prediction
if st.button("Predict Result"):
    # Create the full 17-feature array using "Neutral" averages
    # This matches the structure your model was trained on
    input_data = np.array([
        22.4,        # Age
        1.0,         # Gender
        0.7,         # Ethnicity
        0.8,         # Parental Education
        2.5,         # Family Income
        gpa,         # <--- Your GPA input
        attendance,  # <--- Your Attendance input
        0.6,         # Extracurriculars
        0.4,         # Part-time Job
        1.2,         # Self-study hours
        study_hours, # <--- Your Study Hours input
        4.5,         # History of Failures
        0.8,         # Scholarship
        0.7,         # Internet Access
        stress,      # <--- Your Stress Index input
        2.1,         # Financial Stress
        0.6          # Peer Influence
    ])

    # Reshape for the model (expects a 2D array)
    input_data = input_data.reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display Result
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
    else:
        st.success("### Result: Likely to Stay")

st.info("Note: This model uses neutral averages for features not collected in this form.")