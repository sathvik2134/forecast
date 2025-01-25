# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:11:04 2025

@author: sathw
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
file_path = 'C:\\Users\\sathw\\Downloads\\europeus.csv'# Replace with your file path
data = pd.read_csv(file_path)
data['Time'] = pd.to_datetime(data['Time'])
data.set_index('Time', inplace=True)
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['Close_Lag_1'] = data['Close'].shift(1)
data['Close_Lag_2'] = data['Close'].shift(2)
data['Close_MA_3'] = data['Close'].rolling(window=3).mean()
data['Close_MA_7'] = data['Close'].rolling(window=7).mean()
data['Close_EMA_3'] = data['Close'].ewm(span=3).mean()
data['Close_EMA_7'] = data['Close'].ewm(span=7).mean()
data = data.dropna()

# Split training data (up to 2024) and forecast period (2025)
train_data = data[data.index < '2025-01-01']
forecast_data = data[data.index >= '2025-01-01']

# Features and target
features = ['Open', 'High', 'Low', 'Volume', 'hour', 'day_of_week', 'month',
            'Close_Lag_1', 'Close_Lag_2', 'Close_MA_3', 'Close_MA_7', 'Close_EMA_3', 'Close_EMA_7']
X_train = train_data[features]
y_train = train_data['Close']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
xgb_model = XGBRegressor(tree_method='gpu_hist', random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'learning_rate': [0.01, 0.1, 0.2]
}

xgb_random_search = RandomizedSearchCV(
    xgb_model, param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
)
xgb_random_search.fit(X_train_scaled, y_train)

# Assign the best model
best_rf_model = xgb_random_search.best_estimator_
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])

lstm_model = Sequential([
    LSTM(100, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.3),
    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
from sklearn.preprocessing import MinMaxScaler

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = lstm_model.fit(
    X_train_lstm,  # Input data for training
    y_train_scaled,  # Target data for training
    epochs=50,  # Number of training epochs
    batch_size=64,  # Batch size
    validation_split=0.2,  # Fraction of data used for validation
    callbacks=[early_stop],  # Early stopping callback
    verbose=1  # Verbosity of training logs
)
# Prepare for rolling prediction for 2025
last_known_data = train_data.iloc[-1]
forecast_index = pd.date_range(start='2025-01-01', end='2025-12-31 23:00:00', freq='H')
forecast_results = []

# Rolling prediction
def create_features(last_row, timestamp):
    return {
        'Open': last_row['Open'],
        'High': last_row['High'],
        'Low': last_row['Low'],
        'Volume': last_row['Volume'],
        'hour': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'month': timestamp.month,
        'Close_Lag_1': last_row['Close'],
        'Close_Lag_2': last_row['Close_Lag_1'],
        'Close_MA_3': (last_row['Close'] + last_row['Close_Lag_1'] + last_row['Close_Lag_2']) / 3,
        'Close_MA_7': (last_row['Close'] + last_row['Close_Lag_1'] + last_row['Close_Lag_2'] +
                       last_row.get('Close_Lag_3', last_row['Close']) +
                       last_row.get('Close_Lag_4', last_row['Close']) +
                       last_row.get('Close_Lag_5', last_row['Close']) +
                       last_row.get('Close_Lag_6', last_row['Close'])) / 7,
        'Close_EMA_3': last_row['Close_EMA_3'],
        'Close_EMA_7': last_row['Close_EMA_7']
    }

for timestamp in forecast_index:
    features_for_prediction = create_features(last_known_data, timestamp)
    features_scaled = scaler.transform(pd.DataFrame([features_for_prediction]))
    
    # Predictions from both models
    rf_prediction = best_rf_model.predict(features_scaled)[0]
    lstm_prediction = lstm_model.predict(features_scaled.reshape(1, 1, -1))[0, 0]
    
    # Combine predictions
    prediction = (rf_prediction + lstm_prediction) / 2
    forecast_results.append({'Time': timestamp, 'Predicted_Close': prediction})
    
    # Update last_known_data for next iteration
    last_known_data = {
        **last_known_data,
        'Close': prediction,
        'Close_Lag_1': last_known_data['Close'],
        'Close_Lag_2': last_known_data['Close_Lag_1']
    }
    # Convert forecast results to DataFrame
forecast_df = pd.DataFrame(forecast_results)
forecast_df.set_index('Time', inplace=True)

# Visualization: Monthly breakdown
def predict_forex(input_features):
    """
    Predict the forex closing price based on input features using both XGBoost and LSTM models.

    :param input_features: Dictionary with keys 'date' and 'time'.
                           Example: {'date': '2025-01-27', 'time': '17:00:00'}
    :return: Predicted closing price as a float.
    """
    from datetime import datetime
    import pandas as pd  # Import pandas for DataFrame creation

    # Parse input date and time
    input_date = input_features['date']
    input_time = input_features['time']
    input_datetime = datetime.strptime(f"{input_date} {input_time}", "%Y-%m-%d %H:%M:%S")

    # Generate features for prediction
    features_for_prediction = {
        'Open': last_known_data['Open'],  # Access from last_known_data
        'High': last_known_data['High'],
        'Low': last_known_data['Low'],
        'Volume': last_known_data['Volume'],
        'hour': input_datetime.hour,
        'day_of_week': input_datetime.dayofweek,
        'month': input_datetime.month,
        'Close_Lag_1': last_known_data['Close'],
        'Close_Lag_2': last_known_data['Close_Lag_1'],
        'Close_MA_3': (last_known_data['Close'] + last_known_data['Close_Lag_1'] + last_known_data['Close_Lag_2']) / 3,
        'Close_MA_7': (last_known_data['Close'] + last_known_data['Close_Lag_1'] + last_known_data['Close_Lag_2'] +
                       last_known_data.get('Close_Lag_3', last_known_data['Close']) +
                       last_known_data.get('Close_Lag_4', last_known_data['Close']) +
                       last_known_data.get('Close_Lag_5', last_known_data['Close']) +
                       last_known_data.get('Close_Lag_6', last_known_data['Close'])) / 7,
        'Close_EMA_3': last_known_data['Close_EMA_3'],
        'Close_EMA_7': last_known_data['Close_EMA_7']
    }

    # Create a DataFrame from features
    features_df = pd.DataFrame([features_for_prediction])

    # Scale features using the previously fitted scaler
    features_scaled = scaler.transform(features_df)

    # Make predictions using both models
    rf_prediction = best_rf_model.predict(features_scaled)[0]
    lstm_prediction = lstm_model.predict(features_scaled.reshape(1, 1, -1))[0, 0]

    # Combine predictions (average)
    predicted_price = (rf_prediction + lstm_prediction) / 2

    return float(predicted_price)
