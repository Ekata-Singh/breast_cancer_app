import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🧠 Breast Cancer Classification App")
st.write("Enter the following details to predict whether the tumor is **Malignant (M)** or **Benign (B)**.")

# Feature input (30 features)
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

input_values = []
for feature in features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.4f")
    input_values.append(value)

if st.button("Predict"):
    # Scale input
    input_array = scaler.transform([input_values])

    # Make prediction
    prediction = model.predict(input_array)[0]
    result = "Malignant (M)" if prediction == 1 else "Benign (B)"
    
    st.success(f"### 🎯 Prediction: {result}")
