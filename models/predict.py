import numpy as np

def predict_next_day(model, df):

    features = [
        "Return",
        "MA10",
        "MA20",
        "Volatility",
        "RSI",
        "MACD",
        "MACD_signal",
        "BB_high",
        "BB_low",
        "Momentum",
        "Lag1",
        "Lag2",
        "Lag3"
    ]

    latest_data = df.iloc[-1][features].values.reshape(1, -1)

    predicted_return = model.predict(latest_data)[0]

    current_price = df.iloc[-1]["Price"]

    predicted_price = current_price * (1 + predicted_return)

    if predicted_return > 0.005:
        signal = "BUY"
    elif predicted_return < -0.005:
        signal = "SELL"
    else:
        signal = "HOLD"

    return predicted_return, predicted_price, signal