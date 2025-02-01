# disease_prediction_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("disease_data.csv")

# Preprocess the data
# Assuming the 'diabetes' column is the target variable and others are features
df = df.dropna()  # Drop rows with missing values

# Encode categorical variables like 'gender' and 'smoking_history'
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})  # Encoding gender (0: Male, 1: Female)
df['smoking_history'] = df['smoking_history'].map({'Never smoked': 0, 'Formerly smoked': 1, 'Currently smoking': 2})

# Features and target
X = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['diabetes']  # Target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training with Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions and accuracy check
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
print(f"Model accuracy: {accuracy * 100:.2f}%")


# Save the trained model and scaler to disk
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
