import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def train_lstm(df):

    prices = df["Price"].values.reshape(-1,1)

    # Normalize prices
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X = []
    y = []

    window = 30

    for i in range(window, len(prices_scaled)):
        X.append(prices_scaled[i-window:i])
        y.append(prices_scaled[i])

    X = np.array(X)
    y = np.array(y)

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(32))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    model.fit(
        X,
        y,
        epochs=15,
        batch_size=32,
        verbose=1
    )

    model.save("saved_models/lstm_model.h5")


    joblib.dump(scaler, "saved_models/lstm_scaler.pkl")

    return model, scaler