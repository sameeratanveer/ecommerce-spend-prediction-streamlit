import streamlit as st 
import numpy as np
import pickle 

# Model : 
# Use the saved model : 
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

st.title("E-COMMERCE PREDICTION - yearly amount spent")

# Asking user input. 
avg_session_length = st.number_input("Enter Average Session Length minutes")
time_on_app = st.number_input("Enter time spent on app in Minutes")
length_of_membership = st.number_input("Enter the length of membership in years : ")

# Prediction. 
if st.button(label='PREDICT'):
    data = np.array([avg_session_length,time_on_app,length_of_membership]).reshape(1,-1)
    data_scaled = loaded_scaler.transform(data)
    prediction = loaded_model.predict(data_scaled)
    st.success(f"Prediction Yearly Amount Spent : ${prediction[0]}")

    
