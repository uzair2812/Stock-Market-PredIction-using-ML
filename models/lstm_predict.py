import numpy as np
import joblib
from tensorflow.keras.models import load_model


def predict_lstm(df):

    # Load model and scaler
    model = load_model("saved_models/lstm_model.h5")
    scaler = joblib.load("saved_models/lstm_scaler.pkl")

    prices = df["Price"].values.reshape(-1,1)

    prices_scaled = scaler.transform(prices)

    window = 30

    last_sequence = prices_scaled[-window:]

    X = last_sequence.reshape(1, window, 1)

    predicted_scaled = model.predict(X)

    predicted_price = scaler.inverse_transform(predicted_scaled)

    return predicted_price[0][0]