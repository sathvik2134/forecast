# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:03:36 2025

@author: sathw
"""


import gradio as gr
import importlib.util
from datetime import datetime
import pandas as pd

# Load the integrated `model1.py`
from google.colab import files




# Prediction function for Gradio
def predict_forex(date, time):
    """
    Use the model in `model1.py` to predict the forex closing price for the given date and time.

    :param date: Input date as a string (YYYY-MM-DD)
    :param time: Input time as a string (HH:MM:SS)
    :return: Predicted closing price as a string
    """
    try:
        # Combine date and time into a single datetime object
        input_datetime = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")

        # Prepare input features for the model
        input_features = {
            "date": input_datetime.strftime("%Y-%m-%d"),
            "time": input_datetime.strftime("%H:%M:%S"),
        }

        # Call the prediction function in `model1.py`
        predicted_close = model1.predict_forex(input_features)  # Ensure `predict_forex` exists in `model1.py`

        # Return the prediction
        return f"Predicted Closing Price: {predicted_close:.4f}"
    except AttributeError:
        return "Error: The function `predict_forex` was not found in `model1.py`. Please verify the function name."
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Forex Market Prediction Dashboard")
    gr.Markdown(
        """
        Enter a date and time to predict the closing price of the forex market.
        The prediction uses a model integrated in `model1.py`.
        """
    )

    # Inputs for date and time
    date_input = gr.Text(label="Enter Date (YYYY-MM-DD):", placeholder="e.g., 2025-01-27")
    time_input = gr.Text(label="Enter Time (HH:MM:SS):", placeholder="e.g., 17:00:00")

    # Button to trigger prediction and display result
    predict_button = gr.Button("Predict Closing Price")
    output = gr.Text(label="Prediction Result")

    # Bind button to prediction function
    predict_button.click(predict_forex, inputs=[date_input, time_input], outputs=output)

# Launch the Gradio app
demo.launch()
