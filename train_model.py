import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('student_dropout_dataset_v3.csv')

# Use only your 4 specific features
features = ['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA']
X = df[features]

# Auto-detect target column
target_col = 'Target' if 'Target' in df.columns else 'Dropout'
y = df[target_col]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Print the classes so we know what 0 and 1 mean
print(f"Model Classes: {model.classes_}") 
print("Success! Model trained and saved.")