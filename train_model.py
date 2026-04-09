import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 1. Load data
df = pd.read_csv('student_dropout_dataset_v3.csv')

# 2. Define features and target
features = ['Study_Hours_per_Day', 'Stress_Index', 'Attendance_Rate', 'GPA']
X = df[features]
y = df['Dropout']

# 3. Use 'balanced' class weights to handle data imbalance
# also increase n_estimators to 200 for better learning
# = RandomForestClassifier(
    #n_estimators=200, 
    #max_depth=10, 
    #class_weight='balanced', 
    #random_state=42
)
#model.fit(X, y)
# 3. Advanced Random Forest Configuration
# adding 'min_samples_split' and 'bootstrap' to improve generalization
model = RandomForestClassifier(
    n_estimators=300,        # More trees = more stable predictions
    max_depth=12,            # Allows the model to capture more complex patterns
    min_samples_split=2,     # Minimum data points required to split a node
    class_weight='balanced_subsample', # Better for small, imbalanced datasets
    bootstrap=True,
    random_state=42
)

model.fit(X, y)

# 4. Check accuracy in terminal
score = model.score(X, y)
print(f"New Training Accuracy: {score:.2%}")

# 5. Save the improved model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)