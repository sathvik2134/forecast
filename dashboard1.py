# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 21:50:55 2025

@author: sathw
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import importlib.util

# Load the integrated model1.py
spec = importlib.util.spec_from_file_location("model1", "C:\\Users\\sathw\\model1.py")
model1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model1)

# Streamlit Dashboard
st.title("Forex Market Prediction Dashboard")

# Add a description
st.markdown(
    """
    This dashboard predicts the closing price of the forex market for a given date and time. 
    Enter the desired date and time, and the model will forecast the closing price.
    """
)

# User input for date and time
date_input = st.date_input("Select a Date:", min_value=datetime(2025, 1, 1), max_value=datetime(2025, 12, 31))
time_input = st.time_input("Select a Time:", value=datetime.now().time())

# Trigger prediction
if st.button("Predict Closing Price"):
    # Combine date and time into a single timestamp
    input_datetime = datetime.combine(date_input, time_input)

    # Prepare input for the model
    input_features = {
        "date": input_datetime.strftime("%Y-%m-%d"),  # Example format: '2025-01-27'
        "time": input_datetime.strftime("%H:%M:%S"),  # Example format: '17:00:00'
    }

    try:
        # Call the prediction function from model1.py
        predicted_close = model1.predict_forex(input_features)  # Ensure this function exists in model1.py
        
        # Display the prediction
        st.success(f"Predicted Closing Price: {predicted_close:.4f}")
    except AttributeError:
        st.error("The function 'predict_forex' was not found in model1.py. Please verify the function name.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
