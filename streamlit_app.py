import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

st.title("🎓 Student Dropout Predictor")
st.write("Enter student metrics below to predict enrollment status.")

# 2. User Input Widgets
gpa = st.number_input("Cumulative GPA", 0.0, 4.0, 3.0)
attendance = st.slider("Attendance Percentage", 0, 100, 85)
study_hours = st.number_input("Weekly Study Hours", 0, 100, 10)
stress = st.slider("Stress Index (1-10)", 1, 10, 5)

# 3. Prediction Logic (Triggered by Button)
if st.button("Predict Dropout Status"):
    # We define the array ONLY when the button is pressed
    input_data = np.array([
        22.4, 1.0, 0.7, 0.8, 2.5,  # Hidden averages
        gpa,                       # Your GPA input
        attendance,                # Your Attendance input
        0.6, 0.4, 1.2,             # Hidden averages
        study_hours,               # Your Study Hours input
        4.5, 0.8, 0.7,             # Hidden averages
        stress,                    # Your Stress input
        2.1, 0.6                   # Hidden averages
    ])
    
    # Reshape and Predict
    prediction = model.predict([input_data])
    
    # 4. Display Results
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
        st.warning("Recommendation: Early intervention advised.")
    else:
        st.success("### Result: Likely to Stay")
        st.balloons()