import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from train_code_clasification.train_risk_classifier import build_sequences

model = tf.keras.models.load_model('risk_lstm_model.h5')
with open('scaler_lstm.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
df = pd.read_csv('train_code_clasification/dataset.csv', parse_dates=['Date'])
X, y = build_sequences(df)
X_flat = X.reshape(-1, X.shape[2])
X_scaled = scaler.transform(X_flat).reshape(X.shape)

y_pred = np.argmax(model.predict(X_scaled), axis=1)

print("Classification Report:")
print(classification_report(y, y_pred, target_names=['niski', 'średni', 'wysoki']))

print("Confusion Matrix:")
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['niski', 'średni', 'wysoki'], yticklabels=['niski', 'średni', 'wysoki'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("risk_lstm_confusion_matrix.png", dpi=300)
plt.close()
