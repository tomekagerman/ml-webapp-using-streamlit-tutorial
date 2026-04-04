import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# 2. App Title and Description
st.title("🎓 Student Dropout Predictor")
st.write("Enter the student's details below to predict the likelihood of dropout.")

# 3. Create Inputs (Replacing  HTML form)
st.header("Student Metrics")
col1, col2 = st.columns(2)

with col1:
    gpa = st.number_input("Cumulative GPA", min_value=0.0, max_value=4.0, value=3.0)
    attendance = st.slider("Attendance Percentage", 0, 100, 85)

with col2:
    study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=10)
    stress = st.slider("Stress Index (1-10)", 1, 10, 5)

# 4. Prediction Logic
# Using Neutral Averages so the model doesn't assume "0" for everything else
    input_data = np.array([
        22.4,   # Age
        1.0,    # Gender
        0.7,    # Ethnicity
        0.8,    # Parental Education
        2.5,    # Family Income
        gpa,    # User Input
        attendance, # User Input
        0.6,    # Extracurriculars
        0.4,    # Part-time Job
        1.2,    # Self-study hours
        study_hours, # User Input
        4.5,    # History of Failures
        0.8,    # Scholarship
        0.7,    # Internet Access
        stress, # User Input
        2.1,    # Financial Stress
        0.6     # Peer Influence
    ])
    
    # Ensure the input is the correct shape for the model
    prediction = model.predict([input_data])
    
    # 5. Display Results
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
        st.warning("Recommendation: Early intervention and academic counseling advised.")
    else:
        st.success("### Result: Likely to Stay")
        st.balloons() 

    # 5. The Trigger Button
if st.button("Predict Dropout Status"):
    # Make the prediction using the array you already built
    prediction = model.predict([input_data])
    
    # 6. Display the result
    if prediction[0] == 1:
        st.error("### Result: Likely to Dropout")
    else:
        st.success("### Result: Likely to Stay")