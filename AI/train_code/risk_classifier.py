
from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ────────────────────────────────────────────────────────────────────────────────
# GPU CONFIG
# ────────────────────────────────────────────────────────────────────────────────

def configure_gpu(force_device: str = "AUTO") -> None:
    """Detect and set GPU or CPU for training.

    Parameters
    ----------
    force_device: "CPU", "GPU" or "AUTO".
    """

    gpus = tf.config.list_physical_devices("GPU")
    if force_device.upper() == "CPU":
        tf.config.set_visible_devices([], "GPU")
        print("[INFO] Forced CPU mode – GPU disabled")
        return

    if force_device.upper() == "GPU":
        if not gpus:
            print("[WARN] No GPU available, falling back to CPU")
            return
    # AUTO – use GPU if available
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Detected {len(gpus)} GPU(s), enabled memory growth")
        except RuntimeError as e:
            print("[WARN] Failed to configure GPU:", e)
    else:
        print("[INFO] No GPU found, training on CPU")

# ────────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & LABELING
# ────────────────────────────────────────────────────────────────────────────────

def build_feature_df(csv_dir: Path | str = "data", min_len: int = 250) -> pd.DataFrame:
    csv_dir = Path(csv_dir)
    frames = []

    for csv_file in csv_dir.glob("*.csv"):
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        if len(df) < min_len:
            continue

        df.sort_values("Date", inplace=True)
        df["return"] = df["Close"].pct_change()
        df["abs_return"] = df["return"].abs()
        df["ma5"] = df["return"].rolling(5).mean()
        df["ma20"] = df["return"].rolling(20).mean()
        df["hv20"] = df["return"].rolling(20).std() * math.sqrt(252)

        # Risk categories → codes: -1 (NaN) / 0 / 1 / 2
        risk_cat = pd.cut(df["hv20"], [-np.inf, 0.20, 0.40, np.inf])
        df["risk_label"] = risk_cat.cat.codes  # -1 where hv20 = NaN

        feature_cols = ["return", "abs_return", "ma5", "ma20", "hv20"]
        clean_df = df[df["risk_label"] != -1].dropna(subset=feature_cols)
        frames.append(clean_df[feature_cols + ["risk_label"]])

    if not frames:
        raise RuntimeError("No input data met the criteria – check the CSV directory or run the pipeline.")

    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X = df.drop("risk_label", axis=1).values
    y = df["risk_label"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
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
    device: str = "AUTO",
) -> None:
    configure_gpu(device)

    df = build_feature_df(csv_dir)
    X_train, X_test, y_train, y_test, scaler = prepare_dataset(df)

    model = build_model(X_train.shape[1])

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    model.fit(
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

    print(f"[OK] Saved model to {model_out} and scaler to {scaler_out}")

# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def cli():
    parser = argparse.ArgumentParser(description="Stock risk classification (NN, GPU-ready)")
    parser.add_argument("--csv_dir", default="data", help="Directory with CSV files from the pipeline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["CPU", "GPU", "AUTO"], default="AUTO", help="Force compute device")
    args = parser.parse_args()

    train_and_evaluate(
        csv_dir=args.csv_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
        device=args.device,
    )

if __name__ == "__main__":
    cli()

