from data.data_loader import load_stock_data
from features.feature_engineering import create_features
from models.predict import predict_next_day
from models.lstm_predict import predict_lstm

import joblib


def run_prediction(ticker):

    # Load data
    df = load_stock_data(ticker)

    # Create features
    df_features = create_features(df)

    # Load XGBoost model
    xgb_model = joblib.load("saved_models/xgboost_model.pkl")

    # XGBoost prediction
    xgb_return, xgb_price, signal = predict_next_day(xgb_model, df_features)

    # LSTM prediction
    lstm_price = predict_lstm(df)

    current_price = df.iloc[-1]["Price"]

    # Ensemble price (average of both models)
    final_price = (xgb_price + lstm_price) / 2

    return {
        "current_price": current_price,
        "xgb_return": xgb_return,
        "xgb_price": xgb_price,
        "lstm_price": lstm_price,
        "final_price": final_price,
        "signal": signal
    }