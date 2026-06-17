from data.data_loader import load_stock_data
from features.feature_engineering import create_features
from models.xgboost_model import train_xgboost_model
from models.predict import predict_next_day

import matplotlib.pyplot as plt
import xgboost as xgb


df = load_stock_data("AAPL")

df_features = create_features(df)

model = train_xgboost_model(df_features)

pred_return, pred_price, signal = predict_next_day(model, df_features)

print("Predicted Return:", pred_return)
print("Predicted Price:", pred_price)
print("Signal:", signal)


# Feature Importance Plot
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()