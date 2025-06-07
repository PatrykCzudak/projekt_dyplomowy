import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_other = np.load('X_other.npy')
X_symbol = np.load('X_symbol.npy')
y = np.load('y.npy')

split_idx = int(len(X_other) * 0.8)
X_other_test = X_other[split_idx:]
X_symbol_test = X_symbol[split_idx:]
y_test = y[split_idx:]

model = tf.keras.models.load_model('prediction_model_5d.h5')

y_pred = model.predict({'other_features': X_other_test, 'symbol_code': X_symbol_test}).flatten()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("True vs Predicted 5-day Returns")
plt.xlabel("Sample")
plt.ylabel("Normalized Return")
plt.savefig("cnn_transformer_returns.png", dpi=300)
plt.close()
