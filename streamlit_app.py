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
    # Replicating the 17-feature array logic from Flask code
    input_data = np.array([
        22.4, 1.0, 0.7, 0.8, 2.5,  # Defaults (Age, Gender, etc.)
        gpa, attendance,           # Your inputs
        0.6, 0.4, 1.2,             # Defaults
        study_hours,               # Your input
        4.5, 0.8, 0.7,             # Defaults
        stress,                    # Your input
        2.1, 0.6                   # Defaults
    ])

    prediction = model.predict([input_data])
    
    # Display Result
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
    else:
        st.success("### Result: Likely to Stay")

st.info("Note: This model uses neutral averages for features not collected in this form.")