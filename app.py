import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from data.data_loader import load_stock_data
from features.feature_engineering import create_features


st.set_page_config(layout="wide")
st.title("AI Stock Prediction Dashboard")

ticker = st.text_input("Enter Stock Ticker", "TCS.NS")

if st.button("Run Prediction"):
    df = load_stock_data(ticker)
    if df.empty:
        st.error("No data found for this ticker. Please check the symbol.")
        st.stop()

    # Get company name
    import yfinance as yf
    company_info = yf.Ticker(ticker).info
    company_name = company_info.get("longName", ticker)

    st.markdown(f"## 🏢 Company: {company_name}")
    st.markdown(f"### 📊 Stock Ticker: `{ticker.upper()}`")
    st.divider()

    df_features = create_features(df)
    st.subheader(f"📋 Raw Stock Data of {company_name} (Last 20 rows)")
    st.dataframe(df.tail(20))

    # =========================
    # MOVING AVERAGES
    # =========================
    df["MA10"] = df["Price"].rolling(10).mean()
    df["MA20"] = df["Price"].rolling(20).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["Price"], name="Price"))
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA10"], name="MA10"))
    fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20"))

    st.subheader("Price with Moving Averages")
    st.plotly_chart(fig_ma, use_container_width=True)

    # =========================
    # XGBOOST MODEL (TRAIN EACH TIME)
    # =========================

    features = [
        "Return","MA10","MA20","Volatility","RSI",
        "MACD","MACD_signal","BB_high","BB_low",
        "Momentum","Lag1","Lag2","Lag3"
    ]

    X = df_features[features]
    y = df_features["Target"]

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5
    )

    xgb_model.fit(X_train, y_train)

    preds = xgb_model.predict(X_test)

    actual_prices = df_features["Price"].iloc[split:]
    predicted_prices = actual_prices * (1 + preds)

    st.subheader(f"🤖 XGBoost Model Evaluation — {company_name}")
    xgb_table = pd.DataFrame({
        "Actual Price": actual_prices.tail(15).values,
        "Predicted Price": predicted_prices.tail(15).values,
        "Actual Return": y_test.tail(15).values,
        "Predicted Return": preds[-15:]
    })

    st.dataframe(xgb_table)

    fig_ret = go.Figure()
    fig_ret.add_trace(go.Scatter(y=y_test, name="Actual Return"))
    fig_ret.add_trace(go.Scatter(y=preds, name="Predicted Return"))
    st.plotly_chart(fig_ret, use_container_width=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(y=actual_prices, name="Actual Price"))
    fig_price.add_trace(go.Scatter(y=predicted_prices, name="Predicted Price"))
    st.plotly_chart(fig_price, use_container_width=True)

    # =========================
    # LSTM MODEL (TRAIN EACH TIME)
    # =========================

    st.subheader(f"🧠 LSTM Model Evaluation — {company_name}")
    prices = df["Price"].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    window = 30

    X_lstm = []
    y_lstm = []

    for i in range(window, len(scaled_prices)):
        X_lstm.append(scaled_prices[i-window:i])
        y_lstm.append(scaled_prices[i])

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)

    split_lstm = int(len(X_lstm) * 0.8)

    X_train_lstm = X_lstm[:split_lstm]
    y_train_lstm = y_lstm[:split_lstm]

    X_test_lstm = X_lstm[split_lstm:]
    y_test_lstm = y_lstm[split_lstm:]

    lstm_model = Sequential()

    from tensorflow.keras.layers import Dropout

    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(window,1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))

    lstm_model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    lstm_model.fit(
        X_train_lstm,
        y_train_lstm,
        epochs=15,
        batch_size=32,
        verbose=0
    )

    preds_lstm = lstm_model.predict(X_test_lstm)

    actual_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1,1))
    predicted_lstm = scaler.inverse_transform(preds_lstm)

    lstm_table = pd.DataFrame({
        "Actual Price": actual_lstm.flatten()[-15:],
        "Predicted Price": predicted_lstm.flatten()[-15:]
    })

    st.dataframe(lstm_table)

    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(y=actual_lstm.flatten(), name="Actual Price"))
    fig_lstm.add_trace(go.Scatter(y=predicted_lstm.flatten(), name="Predicted Price"))

    st.plotly_chart(fig_lstm, use_container_width=True)

    # =========================
    # AI PREDICTION FOR TOMORROW
    # =========================

    current_price = df.iloc[-1]["Price"]

    latest_data = df_features.iloc[-1][features].values.reshape(1,-1)

    predicted_return = xgb_model.predict(latest_data)[0]

    xgb_price = current_price * (1 + predicted_return)

    last_sequence = scaled_prices[-window:]
    X_future = last_sequence.reshape(1,window,1)

    predicted_scaled = lstm_model.predict(X_future)

    lstm_price = scaler.inverse_transform(predicted_scaled)[0][0]

    final_price = (xgb_price + lstm_price) / 2

    if predicted_return > 0.005:
        signal = "BUY"
    elif predicted_return < -0.005:
        signal = "SELL"
    else:
        signal = "HOLD"

    st.subheader(f"🎯 AI Prediction for Tomorrow — {company_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Current Price", f"${current_price:.2f}")
    col2.metric("XGBoost Price", f"${xgb_price:.2f}")
    col3.metric("LSTM Price", f"${lstm_price:.2f}")

    st.metric("Final AI Price", f"${final_price:.2f}")

    if signal == "BUY":
        st.success("Signal: BUY")
    elif signal == "SELL":
        st.error("Signal: SELL")
    else:
        st.warning("Signal: HOLD")

    # =========================
    # METRICS
    # =========================

    xgb_rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    xgb_mae = mean_absolute_error(actual_prices, predicted_prices)
    xgb_r2 = r2_score(actual_prices, predicted_prices)

    lstm_rmse = np.sqrt(mean_squared_error(actual_lstm.flatten(), predicted_lstm.flatten()))
    lstm_mae = mean_absolute_error(actual_lstm.flatten(), predicted_lstm.flatten())
    lstm_r2 = r2_score(actual_lstm.flatten(), predicted_lstm.flatten())

    st.subheader(f"📈 Model Performance Comparison — {company_name}")
    col1, col2 = st.columns(2)

    col1.subheader("XGBoost Metrics")
    col1.metric("RMSE", f"{xgb_rmse:.4f}")
    col1.metric("MAE", f"{xgb_mae:.4f}")
    col1.metric("R²", f"{xgb_r2:.4f}")

    col2.subheader("LSTM Metrics")
    col2.metric("RMSE", f"{lstm_rmse:.4f}")
    col2.metric("MAE", f"{lstm_mae:.4f}")
    col2.metric("R²", f"{lstm_r2:.4f}")

    # =========================
    # METRIC GRAPH
    # =========================

    metrics_df = pd.DataFrame({
        "Metric":["RMSE","MAE"],
        "XGBoost":[xgb_rmse,xgb_mae],
        "LSTM":[lstm_rmse,lstm_mae]
    })

    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(x=metrics_df["Metric"],y=metrics_df["XGBoost"],name="XGBoost"))
    fig_metrics.add_trace(go.Bar(x=metrics_df["Metric"],y=metrics_df["LSTM"],name="LSTM"))

    st.plotly_chart(fig_metrics,use_container_width=True)

    # =========================
    # BACKTESTING
    # =========================

    st.subheader(f"💰 Backtesting Simulation — {company_name} (Last 30 Days)")
    backtest_df = df_features.tail(31)

    capital = 10000
    shares = 0

    for i in range(len(backtest_df)-1):

        row = backtest_df.iloc[i]

        features_row = row[features].values.reshape(1,-1)

        predicted_return = xgb_model.predict(features_row)[0]

        price_today = row["Price"]

        if predicted_return > 0.005 and capital > 0:
            shares = capital / price_today
            capital = 0

        elif predicted_return < -0.005 and shares > 0:
            capital = shares * price_today
            shares = 0

    final_value = capital + shares * backtest_df.iloc[-1]["Price"]

    profit = final_value - 10000

    st.metric("Initial Capital", "$10000")
    st.metric("Final Portfolio Value", f"${final_value:.2f}")
    st.metric("Profit/Loss", f"${profit:.2f}")




