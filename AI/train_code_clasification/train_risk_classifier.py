import os
import numpy as np
import pandas as pd
import pickle
import argparse
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

WINDOW_SIZE = 20

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

def compute_features(sub_df):
    sub_df = sub_df.sort_values("Date").reset_index(drop=True)
    sub_df["return"] = sub_df["Close"].pct_change()
    sub_df["abs_return"] = sub_df["return"].abs()
    sub_df["ma5"] = sub_df["Close"].rolling(window=5).mean().pct_change()
    sub_df["ma20"] = sub_df["Close"].rolling(window=20).mean().pct_change()
    sub_df["hv20"] = sub_df["return"].rolling(window=20).std() * np.sqrt(252)
    sub_df["rsi14"] = compute_rsi(sub_df["Close"], period=14)
    sub_df["macd"], sub_df["macd_signal"] = compute_macd(sub_df["Close"])
    return sub_df

def build_sequences(df, future_horizon=1, min_len=250):
    all_X, all_y = [], []
    for symbol, sub_df in df.groupby("Symbol"):
        if len(sub_df) < min_len:
            continue

        sub_df = compute_features(sub_df)
        feature_cols = ["return", "abs_return", "ma5", "ma20", "hv20", "rsi14", "macd", "macd_signal"]
        sub_df = sub_df.dropna(subset=feature_cols)

        # Etykietowanie na podstawie zmienności w przyszłości
        sub_df["risk_label"] = sub_df["hv20"].shift(-future_horizon)
        bins = [-np.inf, 0.15, 0.35, np.inf]
        sub_df["risk_label"] = pd.cut(sub_df["risk_label"], bins=bins, labels=[0, 1, 2])
        sub_df = sub_df.dropna(subset=["risk_label"])
        sub_df["risk_label"] = sub_df["risk_label"].astype(int)

        # Tworzenie rolling windows
        for i in range(len(sub_df) - WINDOW_SIZE - future_horizon):
            window = sub_df.iloc[i:i+WINDOW_SIZE][feature_cols].values
            label = sub_df.iloc[i+WINDOW_SIZE]["risk_label"]
            all_X.append(window)
            all_y.append(label)

    X = np.array(all_X)
    y = np.array(all_y)
    return X, y

def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main(csv_file, future_horizon=1, epochs=20, batch_size=128):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    AI_FOLDER = script_dir
    os.makedirs(os.path.join(AI_FOLDER, "plots"), exist_ok=True)

    print("Wczytuję dane")
    df = pd.read_csv(csv_file, parse_dates=["Date"])

    print("Generuję rolling windows")
    X, y = build_sequences(df, future_horizon=future_horizon)
    print(f"Dane gotowe: X={X.shape}, y={y.shape}")

    num_features = X.shape[2]
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)

    cw = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    class_weights = {i: w for i, w in enumerate(cw)}

    model = build_lstm_model(input_shape=(WINDOW_SIZE, num_features))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]
    history = model.fit(
        X_scaled, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = np.argmax(model.predict(X_scaled), axis=1)
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["niski", "średni", "wysoki"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    model_path = os.path.join(AI_FOLDER, "risk_lstm_model.h5")
    scaler_path = os.path.join(AI_FOLDER, "scaler_lstm.pkl")
    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model zapisano w: {model_path}")
    print(f"Skaler zapisano w: {scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening modelu LSTM dla predykcji ryzyka.")
    parser.add_argument("--csv_file", type=str, required=True, help="Ścieżka do dataset.csv")
    parser.add_argument("--future_horizon", type=int, default=1, help="Horyzont predykcji (w dniach)")
    parser.add_argument("--epochs", type=int, default=20, help="Liczba epok treningu")
    parser.add_argument("--batch_size", type=int, default=128, help="Rozmiar batcha")
    args = parser.parse_args()

    main(
        csv_file=args.csv_file,
        future_horizon=args.future_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size
    )