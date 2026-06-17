from predict_all import run_prediction

result = run_prediction("AAPL")

print("Current Price:", result["current_price"])
print("XGBoost Predicted Return:", result["xgb_return"])
print("XGBoost Predicted Price:", result["xgb_price"])
print("LSTM Predicted Price:", result["lstm_price"])
print("Final Ensemble Price:", result["final_price"])
print("Signal:", result["signal"])