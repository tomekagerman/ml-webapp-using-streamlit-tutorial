import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('student_dropout_dataset_v3.csv')

# Use EXACTLY these 4 features in this EXACT order
features = ['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA']
X = df[features]

# Auto-detect Target column (0=Dropout, 1=Stay)
y = df["Dropout"]
print(y)
# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Success! Model trained on 4 features: {features}")