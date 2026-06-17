from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import numpy as np
import joblib


def train_xgboost_model(df):

    # Features
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

    X = df[features]
    y = df["Target"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("Model RMSE:", rmse)

    joblib.dump(model, "saved_models/xgboost_model.pkl")

    return model

