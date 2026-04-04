import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please ensure it is in the root directory.")

# 2. Set up the UI Header
st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Predictor")
st.markdown("Enter student metrics below to predict enrollment status.")

# 3. Create Input Fields (Matching CSV features)
# Note: Ensure these match the exact features model was trained on
col1, col2 = st.columns(2)

with col1:
    gpa = st.number_input("Cumulative GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
    study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=15)

with col2:
    stress_index = st.slider("Stress Index (1-10)", 1, 10, 5)
    age = st.number_input("Age", min_value=15, max_value=100, value=20)
    gender = st.selectbox("Gender", options=["Male", "Female"])

# 4. Prediction Logic
if st.button("Predict Dropout Status"):
    # Convert inputs into a DataFrame (must match training feature order/names)
    # Mapping categorical 'Gender' to numeric if your model requires it
    gender_val = 1 if gender == "Male" else 0
    
    input_data = pd.DataFrame([[age, gender_val, study_hours, attendance, stress_index, gpa]], 
                              columns=['Age', 'Gender', 'Study_Hours_per_Day', 'Attendance_Rate', 'Stress_Index', 'GPA'])
    
    prediction = model.predict(input_data)
    
    # 5. Display Result
    st.divider()
    if prediction[0] == 1:
        st.error("## Result: Likely to Dropout")
        st.write("Recommendation: Early intervention advised.")
    else:
        st.success("## Result: Likely to Persist")
        st.write("Recommendation: Keep up the good work!")