import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import gradio as gr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv("C:\\Users\\admin\\Downloads\\EURUSD60_cleaned.csv")
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    return df

# Prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

# Build and train LSTM model
def train_lstm(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# Prepare data for Prophet
def prepare_prophet_data(data):
    prophet_df = data[['Close']].reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

# Train Prophet model
def train_prophet(prophet_df):
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    return model

# Prepare data for XGBoost
def prepare_xgboost_data(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data['Close'].values[i-look_back:i])
        y.append(data['Close'].values[i])
    
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test

# Train XGBoost model
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    return model

# Ensemble prediction
def ensemble_predict(lstm_model, prophet_model, xgb_model, data, scaler, input_datetime, look_back=60):
    input_datetime = pd.to_datetime(input_datetime)
    
    # LSTM prediction
    last_sequence = data['Close'].values[-look_back:]
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    scaled_sequence = scaled_sequence.reshape((1, look_back, 1))
    lstm_pred = lstm_model.predict(scaled_sequence)
    lstm_pred = scaler.inverse_transform(lstm_pred)[0][0]
    
    # Prophet prediction
    future = prophet_model.make_future_dataframe(periods=1, freq='H')
    forecast = prophet_model.predict(future)
    prophet_pred = forecast['yhat'].iloc[-1]
    
    # XGBoost prediction
    xgb_input = data['Close'].values[-look_back:].reshape(1, -1)
    xgb_pred = xgb_model.predict(xgb_input)[0]
    
    # Ensemble: Weighted average (50% LSTM, 30% Prophet, 20% XGBoost)
    ensemble_pred = 0.5 * lstm_pred + 0.3 * prophet_pred + 0.2 * xgb_pred
    return ensemble_pred

# Gradio interface
def predict_closing_price(date, time):
    file_path = "EURUSD60_cleaned.csv"
    data = load_data(file_path)
    
    # Prepare and train models
    X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler = prepare_lstm_data(data)
    lstm_model = train_lstm(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    
    prophet_df = prepare_prophet_data(data)
    prophet_model = train_prophet(prophet_df)
    
    X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb = prepare_xgboost_data(data)
    xgb_model = train_xgboost(X_train_xgb, y_train_xgb)
    
    # Combine date and time
    input_datetime = f"{date} {time}"
    
    # Make prediction
    closing_price = ensemble_predict(lstm_model, prophet_model, xgb_model, data, scaler, input_datetime)
    
    # Plot historical data and prediction
    plt.figure(figsize=(10, 6))
    plt.plot(data.index[-100:], data['Close'][-100:], label='Historical Close')
    plt.scatter(pd.to_datetime(input_datetime), closing_price, color='red', label='Predicted Close')
    plt.title('EUR/USD Closing Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return f"Predicted Closing Price: {closing_price:.5f}", plt

# Gradio dashboard
iface = gr.Interface(
    fn=predict_closing_price,
    inputs=[
        gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="2025-05-09"),
        gr.Textbox(label="Time (HH:MM)", placeholder="12:00")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Plot()
    ],
    title="EUR/USD Forex Closing Price Prediction",
    description="Enter a date and time to predict the EUR/USD closing price using an ensemble of LSTM, Prophet, and XGBoost models."
)

# Launch Gradio interface
iface.launch()