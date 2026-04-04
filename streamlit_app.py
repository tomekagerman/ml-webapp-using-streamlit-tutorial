import streamlit as st
import pandas as pd
import pickle

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {e}")

st.set_page_config(page_title="Student Dropout Predictor", page_icon="🎓")
st.title("🎓 Student Dropout Predictor")
st.markdown("Enter student metrics to predict dropout risk.")

# --- INPUT SECTION ---
# Dividing features into 3 columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=15, max_value=50, value=20)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    income = st.number_input("Family Income", min_value=0, value=25000)
    internet = st.selectbox("Internet Access", options=["Yes", "No"])
    study_hours = st.number_input("Study Hours/Day", 0.0, 24.0, 5.0)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)

with col2:
    delay = st.number_input("Assignment Delay Days", 0, 30, 2)
    travel = st.number_input("Travel Time (Mins)", 0, 300, 30)
    job = st.selectbox("Part-Time Job", options=["Yes", "No"])
    scholarship = st.selectbox("Scholarship", options=["Yes", "No"])
    stress = st.slider("Stress Index (1-10)", 1, 10, 5)

with col3:
    gpa = st.number_input("Current GPA", 0.0, 5.0, 3.5)
    sem_gpa = st.number_input("Semester GPA", 0.0, 5.0, 3.4)
    cgpa = st.number_input("CGPA", 0.0, 5.0, 3.6)
    semester = st.number_input("Current Semester", 1, 8, 2)
    dept = st.selectbox("Department", options=["Science", "Arts", "Business", "CS", "Engineering"])
    parent_edu = st.selectbox("Parental Education", options=["High School", "Bachelor", "Master", "PhD"])

# --- PREDICTION LOGIC ---
if st.button("Predict Status"):
    # Convert categorical selections to numeric/boolean (Adjust if your model expects strings)
    # Most standard models expect 0/1 for binary categories
    gender_val = 1 if gender == "Male" else 0
    internet_val = 1 if internet == "Yes" else 0
    job_val = 1 if job == "Yes" else 0
    scholarship_val = 1 if scholarship == "Yes" else 0
    
    # Mapping for Department and Parental Education (Adjust to match training encoding)
    dept_map = {"Science": 0, "Arts": 1, "Business": 2, "CS": 3, "Engineering": 4}
    parent_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

    # CREATE THE DATAFRAME IN THE EXACT ORDER SEEN IN TRAINING
    # Ensure the columns list matches 'model.feature_names_in_' exactly
    input_data = pd.DataFrame([[
        age, gender_val, income, internet_val, study_hours, attendance, 
        delay, travel, job_val, scholarship_val, stress, 
        gpa, sem_gpa, cgpa, semester, dept_map[dept], parent_map[parent_edu]
    ]], columns=[
        'Age', 'Gender', 'Family_Income', 'Internet_Access', 'Study_Hours_per_Day', 
        'Attendance_Rate', 'Assignment_Delay_Days', 'Travel_Time_Minutes', 
        'Part_Time_Job', 'Scholarship', 'Stress_Index', 'GPA', 'Semester_GPA', 
        'CGPA', 'Semester', 'Department', 'Parental_Education'
    ])
    
    # Make Prediction
    prediction = model.predict(input_data)
    
    st.divider()
    if prediction[0] == 1:
        st.error("## Result: High Risk of Dropout")
    else:
        st.success("## Result: Likely to Stay")