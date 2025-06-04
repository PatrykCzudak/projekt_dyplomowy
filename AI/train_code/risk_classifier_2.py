import os
import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period).mean()
    loss = down.rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def build_feature_df(csv_file: str, min_len: int = 250, future_horizon: int = 1) -> pd.DataFrame:

    df = pd.read_csv(csv_file, parse_dates=["Date"])
    frames = []

    for symbol, sub_df in df.groupby("Symbol"):
        if len(sub_df) < min_len:
            continue

        sub_df = sub_df.sort_values("Date").reset_index(drop=True)

        sub_df["return"] = sub_df["Close"].pct_change()
        sub_df["abs_return"] = sub_df["return"].abs()
        sub_df["ma5"] = sub_df["Close"].rolling(window=5).mean().pct_change()
        sub_df["ma20"] = sub_df["Close"].rolling(window=20).mean().pct_change()
        sub_df["hv20"] = sub_df["return"].rolling(window=20).std() * np.sqrt(252)
        sub_df["rsi14"] = compute_rsi(sub_df["Close"], period=14)
        sub_df["macd"], sub_df["macd_signal"] = compute_macd(sub_df["Close"])

        sub_df["risk_label"] = sub_df["hv20"].shift(-future_horizon)
        bins = [-np.inf, 0.15, 0.35, np.inf]
        sub_df["risk_label"] = pd.cut(sub_df["risk_label"], bins=bins, labels=[0, 1, 2])
        sub_df["risk_label"] = sub_df["risk_label"].cat.add_categories([-1]).fillna(-1).astype(int)

        feature_cols = ["return", "abs_return", "ma5", "ma20", "hv20", "rsi14", "macd", "macd_signal"]
        sub_df = sub_df.dropna(subset=feature_cols + ["risk_label"])
        sub_df = sub_df[sub_df["risk_label"] != -1]

        frames.append(sub_df[feature_cols + ["risk_label"]].copy())

    if not frames:
        raise RuntimeError(f"Brak danych spełniających wymagania w {csv_file}")

    result_df = pd.concat(frames, axis=0).reset_index(drop=True)
    return result_df

def build_model(input_dim: int, learning_rate: float = 1e-3, l2_reg: float = 1e-4,
                dropout_rates: tuple = (0.4, 0.3, 0.2)) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, use_bias=False, input_dim=input_dim,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(dropout_rates[0]))
    model.add(tf.keras.layers.Dense(128, use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(dropout_rates[1]))
    model.add(tf.keras.layers.Dense(64, use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(dropout_rates[2]))
    model.add(tf.keras.layers.Dense(3, activation="softmax"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def prepare_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["risk_label"]).values
    y_raw = df["risk_label"].values
    cw = class_weight.compute_class_weight(class_weight="balanced",
                                           classes=np.unique(y_raw),
                                           y=y_raw)
    class_weights = {i: cw_i for i, cw_i in enumerate(cw)}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_cat = tf.keras.utils.to_categorical(y_raw, num_classes=3)
    X_train, X_test, y_train_cat, y_test_cat, y_train_raw, y_test_raw = train_test_split(
        X_scaled, y_cat, y_raw, test_size=test_size, stratify=y_raw, random_state=random_state)
    return X_train, X_test, y_train_cat, y_test_cat, y_train_raw, y_test_raw, scaler, class_weights


# Pipeline treningowy
def train_and_evaluate(csv_file: str, future_horizon: int = 1, test_size: float = 0.2,
                       random_state: int = 42, epochs: int = 50, batch_size: int = 128,
                       patience_es: int = 5, reduce_lr_patience: int = 3):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    AI_FOLDER = script_dir
    os.makedirs(AI_FOLDER, exist_ok=True)
    model_path = os.path.join(AI_FOLDER, "risk_model_best.h5")
    scaler_path = os.path.join(AI_FOLDER, "scaler.pkl")
    log_dir = os.path.join(AI_FOLDER, "logs", "risk_classifier")
    os.makedirs(log_dir, exist_ok=True)

    print("1) Buduję ramkę cech i etykiet z dataset.csv...")
    df = build_feature_df(csv_file, min_len=250, future_horizon=future_horizon)
    print(f"   → Łącznie próbek: {len(df)}")
    print("   → Rozkład etykiet (procentowo):")
    print(df["risk_label"].value_counts(normalize=True))

    X_train, X_test, y_train_cat, y_test_cat, y_train_raw, y_test_raw, scaler, class_weights = prepare_dataset(
        df, test_size=test_size, random_state=random_state)
    input_dim = X_train.shape[1]
    print(f"   → Wektor cech ma wymiar: {input_dim}")

    model = build_model(input_dim)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_es,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=reduce_lr_patience, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    print("5) Rozpoczynam trening...")
    history = model.fit(X_train, y_train_cat,
                        validation_data=(X_test, y_test_cat),
                        epochs=epochs, batch_size=batch_size,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)

    print("6) Ewaluacja na zbiorze testowym...")
    model.load_weights(model_path)
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"   → Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    y_pred_prob = model.predict(X_test)
    y_pred_raw = np.argmax(y_pred_prob, axis=1)
    print("\nClassification report:")
    print(classification_report(y_test_raw, y_pred_raw, target_names=["niski", "średni", "wysoki"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_raw, y_pred_raw))

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n✅ Model zapisano w: {model_path}")
    print(f"   Scaler zapisano w: {scaler_path}")
    print(f"   Logi TensorBoard w: {log_dir}")
    
        # Tworzymy folder na wykresy
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    #Histogram rozkładu klas w danych
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["risk_label"])
    plt.xlabel("Klasa ryzyka")
    plt.ylabel("Liczba próbek")
    plt.title("Rozkład klas ryzyka w zbiorze danych")
    plt.xticks(ticks=[0, 1, 2], labels=["niski", "średni", "wysoki"])
    class_dist_path = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(class_dist_path)
    plt.close()

    #Loss (osobny wykres)
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")
    loss_path = os.path.join(plots_dir, "loss.png")
    plt.savefig(loss_path)
    plt.close()

    #Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")
    acc_path = os.path.join(plots_dir, "accuracy.png")
    plt.savefig(acc_path)
    plt.close()

    #Confusion Matrix
    cm = confusion_matrix(y_test_raw, y_pred_raw)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["niski", "średni", "wysoki"],
                yticklabels=["niski", "średni", "wysoki"])
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywistość")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trening klasyfikatora ryzyka na dataset.csv.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Ścieżka do scalonego pliku CSV (dataset.csv).")
    parser.add_argument("--future_horizon", type=int, default=1,
                        help="Horyzont w przód do etykietowania.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Frakcja danych na test.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Seed dla podziału na train/test.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Liczba epok treningu.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Rozmiar batcha.")
    parser.add_argument("--patience_es", type=int, default=5,
                        help="Patience dla EarlyStopping.")
    parser.add_argument("--reduce_lr_patience", type=int, default=3,
                        help="Patience dla ReduceLROnPlateau.")
    args = parser.parse_args()

    train_and_evaluate(csv_file=args.csv_file,
                       future_horizon=args.future_horizon,
                       test_size=args.test_size,
                       random_state=args.random_state,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       patience_es=args.patience_es,
                       reduce_lr_patience=args.reduce_lr_patience)
