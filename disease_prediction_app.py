import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit interface with doctor symbol 🩺
st.title("Diabetes Prediction App 🩺")

# Collect user inputs for features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, step=1)
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
smoking_history = st.selectbox("Smoking History", ["Never smoked", "Formerly smoked", "Currently smoking"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", min_value=4.0, max_value=15.0, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50.0, max_value=300.0, step=1.0)

# Button to trigger prediction
predict_button = st.button("Predict")

# Function to process the input and make predictions
def predict_disease():
    # Encode inputs before prediction
    gender_encoded = 0 if gender == "Male" else 1
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    smoking_history_encoded = {"Never smoked": 0, "Formerly smoked": 1, "Currently smoking": 2}[smoking_history]

    # Prepare input for prediction
    input_data = np.array([[gender_encoded, age, hypertension_encoded, heart_disease_encoded,
                            smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict disease status
    prediction = model.predict(input_data_scaled)

    # Output prediction result and display appropriate image
    if prediction == 1:
        st.subheader("Prediction: Diabetes Positive")
        st.image("diabetes_image.jpg", caption="Diabetes Condition")  # Show image for diabetes positive
    else:
        st.subheader("Prediction: Diabetes Negative")
        st.image("no_diabetes_image.jpg", caption="No Diabetes Condition")  # Show image for diabetes negative

    # Optionally, show prediction probability
    probability = model.predict_proba(input_data_scaled)
    st.write(f"Prediction Probability (Positive): {probability[0][1]:.2f}")

# Trigger prediction when button is clicked
if predict_button:
    predict_disease()


# jai mata di
