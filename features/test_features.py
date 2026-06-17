from data.data_loader import load_stock_data
from features.feature_engineering import create_features


df = load_stock_data("AAPL")

df_features = create_features(df)

print(df_features.head())
print(df_features.columns)