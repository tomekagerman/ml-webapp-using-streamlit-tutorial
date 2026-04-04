import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('student_dropout_dataset_v3.csv')

# Step 1: Select ONLY the 4 features you want
features = ['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA']
X = df[features]

# Step 2: Auto-detect the Target column 
# (Checks for 'Target' first, then 'Dropout')
target_col = 'Target' if 'Target' in df.columns else 'Dropout'
y = df[target_col]

# Step 3: Train and Save
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Success! Model trained using {target_col} and saved as model.pkl")