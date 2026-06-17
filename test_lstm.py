from data.data_loader import load_stock_data
from models.lstm_model import train_lstm
from models.lstm_predict import predict_lstm

# Load stock data
df = load_stock_data("AAPL")

# Train model
model, scaler = train_lstm(df)

# Predict next price
pred_price = predict_lstm(df)

print("LSTM Predicted Price:", pred_price)