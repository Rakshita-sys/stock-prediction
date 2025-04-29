from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


lr_model = joblib.load('lr_model.joblib')
lstm_model = load_model('lstm_model.h5')

scaler = MinMaxScaler()

def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data['Close']

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
    return np.array(X)

@app.route('/')
def home():
    return 'Use POST /predict to make predictions.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    start_date = data['start_date']
    end_date = data['end_date']
    
    stock_data = get_stock_data(ticker, start_date, end_date)
    scaled_data = scaler.fit_transform(stock_data.values.reshape(-1, 1))
    
    
    sequence_length = 10
    X = create_sequences(scaled_data, sequence_length)
    X_lr = X[:, -1].reshape(-1, 1)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    
    
    lr_pred = lr_model.predict(X_lr)
    lstm_pred = lstm_model.predict(X_lstm)
    
    
    lr_pred = scaler.inverse_transform(lr_pred.reshape(-1, 1))
    lstm_pred = scaler.inverse_transform(lstm_pred)
    
    
    response = {
        'dates': stock_data.index.strftime('%Y-%m-%d').tolist(),
        'actual': stock_data.values.tolist(),
        'lr_predictions': lr_pred.flatten().tolist(),
        'lstm_predictions': lstm_pred.flatten().tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)