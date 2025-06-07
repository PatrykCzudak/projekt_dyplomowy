import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from train_code_price_prediction.train_bayesian_lstm import create_dataset, compute_rsi, compute_macd

model = tf.keras.models.load_model('price_forecast_bayesian_lstm.h5', compile=False)

with open('price_forecast_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df = pd.read_csv('train_code_price_prediction/dataset.csv', parse_dates=['Date'])
X, y_true = create_dataset(df)
X_flat = X.reshape(-1, X.shape[2])
X_scaled = scaler.transform(X_flat).reshape(X.shape)

y_pred = model.predict(X_scaled)
mu_pred = y_pred[:, :y_true.shape[1]]

mae = mean_absolute_error(y_true, mu_pred)
mse = mean_squared_error(y_true, mu_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_true.flatten(), label='True')
plt.plot(mu_pred.flatten(), label='Predicted')
plt.legend()
plt.title("True vs Predicted Log-Returns (Bayesian LSTM)")
plt.xlabel("Sample")
plt.ylabel("Log-Return")
plt.savefig("bayesian_lstm_logreturns.png", dpi=300)
plt.close()
