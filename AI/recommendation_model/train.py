import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
from pathlib import Path

csv_path = Path('dataset.csv')
df = pd.read_csv(csv_path, parse_dates=['Date'])
df = df.sort_values(['Symbol', 'Date'])

df['Return'] = df.groupby('Symbol')['Close'].pct_change()
df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x / x.shift(20) - 1)
df = df.dropna(subset=['Return', 'Momentum'])

df['Future_Return'] = df.groupby('Symbol')['Return'].shift(-1)
df = df.dropna(subset=['Future_Return'])

features = ['Return', 'Momentum']
X = df[features].values
y = df['Future_Return'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
reg.fit(X_train, y_train)

with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(reg, f)
with open('recommendation_model_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
