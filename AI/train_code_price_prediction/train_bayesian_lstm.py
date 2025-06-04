"""
Trenuje model Bayesian LSTM na rolling window i przewiduje log-zwroty (lub ceny) na 5 dni w przód.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Parametry
WINDOW_SIZE = 20
FORECAST_HORIZON = 5
FEATURE_COLS = ["return", "abs_return", "ma5", "ma20", "hv20", "rsi14", "macd", "macd_signal"]

tfd = tfp.distributions

def build_bayesian_lstm(input_shape, output_size):
    """
    Buduje model Bayesian LSTM z outputem: Normal (mu, sigma)
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size * 2)  # mu + sigma dla każdego kroku
    ])
    return model

def negloglik(y_true, y_pred):
    """
    Negative Log-Likelihood Loss dla modelu probabilistycznego.
    """
    n_outputs = y_true.shape[1]
    mu = y_pred[:, :n_outputs]
    sigma = tf.nn.softplus(y_pred[:, n_outputs:]) + 1e-6
    dist = tfd.Normal(loc=mu, scale=sigma)
    return -tf.reduce_mean(dist.log_prob(y_true))

def create_dataset(df, forecast_horizon=FORECAST_HORIZON):
    """
    Tworzy rolling window (X) i etykiety (y).
    """
    all_X, all_y = [], []

    for symbol, sub_df in df.groupby("Symbol"):
        sub_df = sub_df.sort_values("Date").reset_index(drop=True)
        sub_df["log_close"] = np.log(sub_df["Close"])
        sub_df["return"] = sub_df["Close"].pct_change()
        sub_df["abs_return"] = sub_df["return"].abs()
        sub_df["ma5"] = sub_df["Close"].rolling(window=5).mean().pct_change()
        sub_df["ma20"] = sub_df["Close"].rolling(window=20).mean().pct_change()
        sub_df["hv20"] = sub_df["return"].rolling(window=20).std() * np.sqrt(252)
        sub_df["rsi14"] = compute_rsi(sub_df["Close"])
        sub_df["macd"], sub_df["macd_signal"] = compute_macd(sub_df["Close"])

        sub_df = sub_df.dropna(subset=FEATURE_COLS + ["log_close"])

        for i in range(len(sub_df) - WINDOW_SIZE - forecast_horizon):
            window = sub_df.iloc[i:i+WINDOW_SIZE][FEATURE_COLS].values
            future_prices = sub_df.iloc[i+WINDOW_SIZE:i+WINDOW_SIZE+forecast_horizon]["log_close"].values
            current_price = sub_df.iloc[i+WINDOW_SIZE-1]["log_close"]
            log_returns = future_prices - current_price  # log-return(y)
            all_X.append(window)
            all_y.append(log_returns)

    return np.array(all_X), np.array(all_y)

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

def main(csv_file, epochs=30, batch_size=128):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    AI_FOLDER = script_dir
    os.makedirs(os.path.join(AI_FOLDER, "plots"), exist_ok=True)

    # Wczytaj dane
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    # Rolling window
    X, y = create_dataset(df)
    print(f"✅ Dane przygotowane: X={X.shape}, y={y.shape}")

    # Skalowanie
    num_features = X.shape[2]
    scaler = StandardScaler()
    X_flat = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

    # Train/Test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Model
    model = build_bayesian_lstm(input_shape=(WINDOW_SIZE, num_features), output_size=FORECAST_HORIZON)
    model.compile(optimizer="adam", loss=negloglik)
    model.summary()

    # Trening
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Zapisz model i scaler
    model.save(os.path.join(AI_FOLDER, "price_forecast_bayesian_lstm.h5"))
    with open(os.path.join(AI_FOLDER, "price_forecast_scaler.pkl"), "wb") as f:
        import pickle
        pickle.dump(scaler, f)
    print("✅ Model i scaler zapisane!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trenuje Bayesian LSTM do prognozowania cen.")
    parser.add_argument("--csv_file", type=str, required=True, help="Ścieżka do dataset.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    main(args.csv_file, epochs=args.epochs, batch_size=args.batch_size)
