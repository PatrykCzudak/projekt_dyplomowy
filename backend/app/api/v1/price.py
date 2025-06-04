from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
import pickle
from datetime import timedelta

router = APIRouter()
tfd = tfp.distributions

# Globalne zmienne
price_forecast_model = tf.keras.models.load_model("AI/price_forecast_bayesian_lstm.h5", compile=False)
with open("AI/price_forecast_scaler.pkl", "rb") as f:
    price_forecast_scaler = pickle.load(f)

WINDOW_SIZE = 20
FORECAST_HORIZON = 5
FEATURE_COLS = ["return", "abs_return", "ma5", "ma20", "hv20", "rsi14", "macd", "macd_signal"]

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

@router.get("/price/asset/{symbol}/forecast")
async def forecast_price(symbol: str):
    try:
        df = yf.download(symbol, period="5y", progress=False)
        if df.empty or len(df) < WINDOW_SIZE:
            raise HTTPException(status_code=404, detail="Brak wystarczających danych historycznych.")

        tmp = df["Close"]
        returns = tmp.pct_change().fillna(0).squeeze()
        abs_return = returns.abs()
        ma5 = tmp.rolling(window=5).mean().pct_change().fillna(0).squeeze()
        ma20 = tmp.rolling(window=20).mean().pct_change().fillna(0).squeeze()
        hv20 = returns.rolling(window=20).std().fillna(0).squeeze()
        rsi14 = compute_rsi(tmp, period=14).fillna(0).squeeze()
        macd, macd_signal = compute_macd(tmp)
        macd = macd.fillna(0).squeeze()
        macd_signal = macd_signal.fillna(0).squeeze()

        df_feat = pd.DataFrame({
            "return": returns,
            "abs_return": abs_return,
            "ma5": ma5,
            "ma20": ma20,
            "hv20": hv20,
            "rsi14": rsi14,
            "macd": macd,
            "macd_signal": macd_signal
        }).fillna(0)

        if len(df_feat) < WINDOW_SIZE:
            raise HTTPException(status_code=400, detail="Za mało danych do utworzenia sekwencji.")

        last_window = df_feat.iloc[-WINDOW_SIZE:].to_numpy(dtype=np.float32)
        scaled = price_forecast_scaler.transform(last_window)
        scaled = scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLS))

        # Model Bayesian
        y_pred = price_forecast_model(scaled, training=False)
        mu = y_pred[:, :FORECAST_HORIZON].numpy().flatten()
        sigma = tf.nn.softplus(y_pred[:, FORECAST_HORIZON:]).numpy().flatten() + 1e-6

        # Odtwórz ceny
        last_close = tmp.dropna().iloc[-1]
        forecast = []
        forecast_dates = pd.date_range(start=tmp.index[-1] + timedelta(days=1), periods=FORECAST_HORIZON, freq="B")

        for i in range(FORECAST_HORIZON):
            pred_price = last_close * np.exp(mu[i])
            forecast.append({
                "date": str(forecast_dates[i].strftime('%Y-%m-%dT%H:%M:%SZ')),
                "mean": float(pred_price),
                "std": float(pred_price * sigma[i])
            })

        return {"forecast": forecast}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd w predykcji: {e}")
