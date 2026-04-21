import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("ridge_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🔥 Forest Fire Prediction App")

st.write("Enter the values below:")

# Example input fields (modify based on your dataset columns)
Temperature = st.number_input("Temperature")
RH = st.number_input("Relative Humidity")
Ws = st.number_input("Wind Speed")
Rain = st.number_input("Rain")
FFMC = st.number_input("FFMC")
DMC = st.number_input("DMC")
ISI = st.number_input("ISI")
Classes = st.number_input("Classes (0 or 1)")
Region = st.number_input("Region (0 or 1)")

# Prediction button
if st.button("Predict"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    
    # Scale input
    scaled_data = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(scaled_data)

    st.success(f"Prediction: {prediction[0]}")