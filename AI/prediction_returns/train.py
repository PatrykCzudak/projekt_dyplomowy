import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

csv_path = Path('dataset.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values(['Symbol', 'Date'])

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period).mean()
    loss = down.rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

df['Return'] = df.groupby('Symbol')['Close'].pct_change()
df['RSI'] = df.groupby('Symbol')['Close'].transform(compute_rsi)
df['MACD'] = df.groupby('Symbol')['Close'].transform(lambda x: compute_macd(x)[0])
df['MACD_signal'] = df.groupby('Symbol')['Close'].transform(lambda x: compute_macd(x)[1])
df['Volume'] = df['Volume']

feature_cols = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'MACD_signal']
df = df.dropna(subset=feature_cols)

scaler = MinMaxScaler()
for col in feature_cols:
    df[f'Scaled_{col}'] = df.groupby('Symbol')[col].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())

SEQ_LEN = 20
def create_sequences(data, features):
    X, y = [], []
    for i in range(len(data) - SEQ_LEN):
        window = data.iloc[i:i+SEQ_LEN][features].values
        target = data.iloc[i+SEQ_LEN]['Return']
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

X, y = [], []
for symbol in df['Symbol'].unique():
    df_sym = df[df['Symbol'] == symbol]
    features = [f'Scaled_{col}' for col in feature_cols]
    Xi, yi = create_sequences(df_sym, features)
    X.append(Xi)
    y.append(yi)

X = np.concatenate(X)
y = np.concatenate(y)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, len(feature_cols))),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

model.save('prediction_model.h5')
with open('prediction_model_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
