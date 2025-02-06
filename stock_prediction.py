import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data['Close']

# Fetch data
ticker = 'AAPL'  # Apple Inc. as an example
start_date = '2010-01-01'
end_date = '2023-01-01'
data = get_stock_data(ticker, start_date, end_date)

# Preprocess data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(scaled_data, sequence_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train[:, -1].reshape(-1, 1), y_train)
lr_pred = lr_model.predict(X_test[:, -1].reshape(-1, 1))

# LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train_lstm, y_train, batch_size=64, epochs=20, validation_split=0.1, verbose=1)

lstm_pred = lstm_model.predict(X_test_lstm)

# Inverse transform predictions
lr_pred = scaler.inverse_transform(lr_pred.reshape(-1, 1))
lstm_pred = scaler.inverse_transform(lstm_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE and R2 score
lr_rmse = np.sqrt(mean_squared_error(y_test_actual, lr_pred))
lr_r2 = r2_score(y_test_actual, lr_pred)

lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
lstm_r2 = r2_score(y_test_actual, lstm_pred)

print(f"Linear Regression - RMSE: {lr_rmse}, R2: {lr_r2}")
print(f"LSTM - RMSE: {lstm_rmse}, R2: {lstm_r2}")

# Save models
import joblib
joblib.dump(lr_model, 'lr_model.joblib')
lstm_model.save('lstm_model.h5')

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(lr_pred, label='Linear Regression')
plt.plot(lstm_pred, label='LSTM')
plt.legend()
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.savefig('predictions.png')
plt.close()