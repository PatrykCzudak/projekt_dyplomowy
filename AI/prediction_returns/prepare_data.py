import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import pickle

# Dane
csv_path = Path('dataset.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values(['Symbol', 'Date'])

# Tworzenie Ficzrów
# RSI (Relative Strength Index).
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period).mean()
    loss = down.rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
# MACD (Moving Average Convergence Divergence) 
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
df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=5))
df['MA5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
df['STD5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).std())
df['Return_5d'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=5).shift(-5))
df['Symbol_Code'] = df['Symbol'].astype('category').cat.codes

feature_cols = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'MACD_signal', 'Momentum', 'MA5', 'STD5']
df = df.dropna(subset=feature_cols + ['Symbol_Code', 'Return_5d'])

df = df.reset_index(drop=True)

# Skaler na przedział (0-1)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[feature_cols])
scaled_df = pd.DataFrame(scaled_features, columns=[f'Scaled_{col}' for col in feature_cols], index=df.index)
df = pd.concat([df, scaled_df], axis=1)

scaler_return = StandardScaler()
y_scaled = scaler_return.fit_transform(df['Return_5d'].values.reshape(-1,1)).flatten()

# Sekwencja do uczenia
SEQ_LEN = 20
features_scaled = [f'Scaled_{col}' for col in feature_cols]

# Sekwencja danych wejściowych i wartości docelowych.
def create_sequences(data, target):
    X_other, X_symbol, y = [], [], []
    for i in range(len(data) - SEQ_LEN):
        window_other = data.iloc[i:i+SEQ_LEN][features_scaled].values
        window_symbol = data.iloc[i:i+SEQ_LEN]['Symbol_Code'].values.reshape(-1, 1)
        target_value = target[i+SEQ_LEN]
        X_other.append(window_other)
        X_symbol.append(window_symbol)
        y.append(target_value)
    return np.array(X_other), np.array(X_symbol), np.array(y)

# Grupowanie danych według symbolu i tworzenie sekwencji
X_other, X_symbol, y = [], [], []
for symbol in df['Symbol'].unique():
    df_sym = df[df['Symbol'] == symbol]
    target_scaled = y_scaled[df_sym.index]
    Xi_other, Xi_symbol, yi = create_sequences(df_sym, target_scaled)
    if len(Xi_other) > 0:
        X_other.append(Xi_other)
        X_symbol.append(Xi_symbol)
        y.append(yi)

X_other = np.concatenate(X_other)
X_symbol = np.concatenate(X_symbol)
y = np.concatenate(y)

# Save
np.save('X_other.npy', X_other)
np.save('X_symbol.npy', X_symbol)
np.save('y.npy', y)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('scaler_return.pkl', 'wb') as f:
    pickle.dump(scaler_return, f)

print("Dane przygotowane i zapisane.")
