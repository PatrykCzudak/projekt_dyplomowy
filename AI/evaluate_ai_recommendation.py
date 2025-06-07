import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Wczytaj pipeline
with open('recommendation_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Załaduj dane
csv_path = Path('recommendation_model/dataset.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values(['Symbol', 'Date'])

# Feature engineering (jak w train_ai_recommendation.py)
df['Return'] = df.groupby('Symbol')['Close'].pct_change()
df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x / x.shift(20) - 1)
df['Return_lag1'] = df.groupby('Symbol')['Return'].shift(1)
df['Return_lag5'] = df.groupby('Symbol')['Return'].shift(5)
df['RollingMean_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).mean())
df['RollingStd_5'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5).std())
exp12 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
exp26 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['MACD'] = exp12 - exp26
df['Bollinger_Upper'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean() + 2 * x.rolling(window=20).std())
df['Bollinger_Lower'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean() - 2 * x.rolling(window=20).std())
df['Bollinger_Width'] = df['Bollinger_Upper'] - df['Bollinger_Lower']

def stochastic_oscillator(high, low, close, k=14):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    return 100 * (close - lowest_low) / (highest_high - lowest_low)

df['Stochastic_14'] = df.groupby('Symbol').apply(
    lambda group: stochastic_oscillator(group['Close'], group['Close'], group['Close'], 14)
).reset_index(level=0, drop=True)

df['Volatility_20'] = df.groupby('Symbol')['Return'].transform(lambda x: x.rolling(window=20).std())
features = ['Return', 'Momentum', 'Return_lag1', 'Return_lag5', 'RollingMean_5', 'RollingStd_5',
            'MACD', 'Bollinger_Width', 'Stochastic_14', 'Volatility_20']

df = df.dropna(subset=features)
df['Future_Return_5d'] = df.groupby('Symbol')['Return'].shift(-1).rolling(window=5).mean()
df['Target'] = (df['Future_Return_5d'] > 0).astype(int)
df = df.dropna(subset=['Target'])

features_num = features
features_cat = ['Symbol']
X = df[features_num + features_cat]
y = df['Target']

# Podział na train/test (np. 80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Predykcje
y_pred = pipeline.predict(X_test)

# Raport
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Spadek', 'Wzrost']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spadek', 'Wzrost'], yticklabels=['Spadek', 'Wzrost'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("catboost_confusion_matrix.png", dpi=300)
plt.close()
print("[INFO] Wykres zapisano jako 'catboost_confusion_matrix.png'.")
