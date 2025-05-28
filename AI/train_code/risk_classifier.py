from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ────────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & LABELING
# ────────────────────────────────────────────────────────────────────────────────

def build_feature_df(csv_dir: Path | str = "data", min_len: int = 250) -> pd.DataFrame:
    """Wczytaj wszystkie pliki CSV w katalogu i zbuduj DataFrame z cechami + label."""

    csv_dir = Path(csv_dir)
    frames = []

    for csv_file in csv_dir.glob("*.csv"):
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        if len(df) < min_len:
            continue  # pomijamy zbyt krótkie historie

        df.sort_values("Date", inplace=True)
        df["return"] = df["Close"].pct_change()
        df["abs_return"] = df["return"].abs()
        df["ma5"] = df["return"].rolling(5).mean()
        df["ma20"] = df["return"].rolling(20).mean()
        df["hv20"] = df["return"].rolling(20).std() * math.sqrt(252)

        # etykieta ryzyka
        df["risk_label"] = pd.cut(
            df["hv20"],
            bins=[-np.inf, 0.20, 0.40, np.inf],
            labels=[0, 1, 2],
        ).astype(int)

        # zachowujemy tylko potrzebne kolumny; usuwamy początkowe NaN‑y
        feature_cols = ["return", "abs_return", "ma5", "ma20", "hv20"]
        frames.append(df.dropna(subset=feature_cols + ["risk_label"])[feature_cols + ["risk_label"]])

    if not frames:
        raise RuntimeError("Brak danych wejściowych spełniających kryteria – uruchom yahoo_pipeline.py lub sprawdź ścieżkę.")

    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Podziel na train/test i przeskaluj cechy."""

    X = df.drop("risk_label", axis=1).values
    y = df["risk_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # one‑hot encoding labeli (3 klasy)
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    return X_train_scaled, X_test_scaled, y_train_cat, y_test_cat, scaler

# ────────────────────────────────────────────────────────────────────────────────
# MODEL
# ────────────────────────────────────────────────────────────────────────────────

def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(128, activation="relu", input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ────────────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ────────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    csv_dir: Path | str,
    epochs: int = 30,
    batch_size: int = 256,
    patience: int = 5,
    model_out: str | Path = "risk_model.h5",
    scaler_out: str | Path = "scaler.pkl",
) -> None:
    df = build_feature_df(csv_dir)
    X_train, X_test, y_train, y_test, scaler = prepare_dataset(df)

    model = build_model(X_train.shape[1])

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")

    model.save(model_out)
    with open(scaler_out, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[OK] Zapisano model → {model_out} oraz scaler → {scaler_out}")

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def cli():
    parser = argparse.ArgumentParser(description="Klasyfikacja ryzyka akcji (NN)")
    parser.add_argument("--csv_dir", default="data", help="Katalog z plikami CSV od pipeline'u")
    parser.add_argument("--epochs", type=int, default=30, help="Liczba epok treningu")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    args = parser.parse_args()

    train_and_evaluate(
        csv_dir=args.csv_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
    )

if __name__ == "__main__":
    cli()