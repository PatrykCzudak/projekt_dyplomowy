import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import logging

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select, func, case

from app.db.database import get_db
from app.models.models import Asset, Transaction
from app.services import risk_analysis as ra

router = APIRouter(prefix="/risk", tags=["risk"])

logger = logging.getLogger("uvicorn")

WINDOW_SIZE = 20 

#RSI i MACD
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


#Budowa Cech
def extract_features_for_prediction_yahoo(symbol: str, lookback: int = 21) -> np.ndarray:
    """
    1) Pobiera z Yahoo Finance dane 'Close' za ostatnie 5 lat (5y).
    2) Zapewnia, że mamy pd.Series (jeśli to DataFrame z jedną kolumną, bierze pierwszą).
    3) Na tej serii liczy wskaźniki: return, abs_return, ma5, ma20, hv20, rsi14, macd, macd_signal.
    4) Zwraca wektor (1×8) cech z ostatniego wiersza.
    """
    try:
        raw = yf.download(symbol, period="5y", progress=False)["Close"]
    except Exception:
        raise FileNotFoundError(f"Nie udało się pobrać danych z Yahoo dla symbolu {symbol}")

    if isinstance(raw, pd.DataFrame):
        tmp = raw.iloc[:, 0].dropna()
    else:
        tmp = raw.dropna()

    returns = tmp.pct_change().fillna(0)
    abs_return = returns.abs()
    ma5 = tmp.rolling(window=5).mean().fillna(method="bfill")
    ma20 = tmp.rolling(window=20).mean().fillna(method="bfill")
    hv20 = returns.rolling(window=20).std().fillna(method="bfill")
    rsi14 = compute_rsi(tmp, period=14).fillna(method="bfill")
    macd, macd_signal = compute_macd(tmp)
    macd = macd.fillna(method="bfill")
    macd_signal = macd_signal.fillna(method="bfill")

    df = pd.DataFrame({
        "return":      returns,
        "abs_return":  abs_return,
        "ma5":         ma5,
        "ma20":        ma20,
        "hv20":        hv20,
        "rsi14":       rsi14,
        "macd":        macd,
        "macd_signal": macd_signal
    })

    last_row = df.iloc[-1]

    if last_row.isna().any():
        raise ValueError(f"Niektóre cechy dla {symbol} są NaN – sprawdź dane historyczne")

    return last_row.to_numpy(dtype=np.float32).reshape(1, -1)

def extract_sequence_for_prediction_yahoo(symbol: str) -> np.ndarray:
    """
    Pobiera dane z Yahoo Finance dla tickera i zwraca rolling window (20 dni).
    """
    try:
        raw = yf.download(symbol, period="5y", progress=False)["Close"]
    except Exception:
        raise FileNotFoundError(f"Nie udało się pobrać danych z Yahoo dla symbolu {symbol}")

    if isinstance(raw, pd.DataFrame):
        tmp = raw.iloc[:, 0].dropna()
    else:
        tmp = raw.dropna()

    returns = tmp.pct_change().fillna(0)
    abs_return = returns.abs()
    ma5 = tmp.rolling(window=5).mean().pct_change().fillna(0)
    ma20 = tmp.rolling(window=20).mean().pct_change().fillna(0)
    hv20 = returns.rolling(window=20).std().fillna(0)
    rsi14 = compute_rsi(tmp, period=14).fillna(0)
    macd, macd_signal = compute_macd(tmp)
    macd = macd.fillna(0)
    macd_signal = macd_signal.fillna(0)

    df = pd.DataFrame({
        "return": returns,
        "abs_return": abs_return,
        "ma5": ma5,
        "ma20": ma20,
        "hv20": hv20,
        "rsi14": rsi14,
        "macd": macd,
        "macd_signal": macd_signal
    })

    if len(df) < WINDOW_SIZE:
        raise ValueError(f"Zbyt mało danych ({len(df)}) do utworzenia sekwencji {WINDOW_SIZE}-dniowej")

    seq = df.iloc[-WINDOW_SIZE:].to_numpy(dtype=np.float32).reshape(1, WINDOW_SIZE, -1)
    return seq

#Wczytanie modelu AI i scalera przy starcie
risk_model: tf.keras.Model | None = None
risk_scaler = None

@router.on_event("startup")
async def load_risk_model_event():
    """
    Podczas startu FastAPI wczytujemy model i scaler z katalogu AI/.
    """
    global risk_model, risk_scaler

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(current_dir, "../../../.."))
    ai_folder = os.path.join(project_root, "AI")

    model_path = os.path.join(ai_folder, "risk_lstm_model.h5")
    scaler_path = os.path.join(ai_folder, "scaler_lstm.pkl")

    if not os.path.exists(model_path):
        print(f"[WARN] Nie znaleziono modelu pod {model_path}")
        return
    if not os.path.exists(scaler_path):
        print(f"[WARN] Nie znaleziono scalera pod {scaler_path}")
        return

    try:
        risk_model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            risk_scaler = pickle.load(f)
        print(f"[OK] Załadowano model ryzyka AI z {model_path}")
    except Exception as e:
        print(f"[ERROR] Błąd podczas wczytywania modelu/scalera: {e}")


#Modele odpowiedzi dla AI
class AssetRiskResponse(BaseModel):
    risk_category: str
    risk_prediction: float 

class PortfolioRiskResponse(BaseModel):
    portfolio_id: int
    risk_category: str
    risk_prediction: float



#Bieżące pozycje portfela z tabeli Transaction
def get_current_portfolio_positions(db: Session) -> list[tuple[str, float]]:
    """
    Zwraca listę krotek (symbol, net_qty) tylko dla tych assetów,
    w których net_qty = SUM(BUY.quantity) - SUM(SELL.quantity) > 0.
    """
    qty_case = case(
        (Transaction.type == "BUY",  Transaction.quantity),
        (Transaction.type == "SELL", -Transaction.quantity),
        else_=0.0
    )

    stmt = (
        select(
            Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(Transaction, Asset.id == Transaction.asset_id)
        .group_by(Asset.id, Asset.symbol)
        .having(func.sum(qty_case) > 0)
    )

    rows = db.execute(stmt).all()
    return [(row.symbol, float(row.net_qty)) for row in rows]


#Endpoint: AI Asset Risk
@router.get("/asset/{symbol}/ai", response_model=AssetRiskResponse)
async def get_asset_risk_ai(symbol: str):
    global risk_model, risk_scaler
    if risk_model is None or risk_scaler is None:
        raise HTTPException(status_code=503, detail="Model AI nie jest jeszcze dostępny")

    try:
        features = extract_sequence_for_prediction_yahoo(symbol.upper())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Brak danych historycznych dla {symbol.upper()}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    num_features = features.shape[2]
    scaled = features.reshape(-1, num_features)
    scaled = risk_scaler.transform(scaled)
    scaled = scaled.reshape(1, WINDOW_SIZE, num_features)

    probs = risk_model.predict(scaled).flatten().tolist()
    pred_raw = int(np.argmax(probs))
    labels = ["niski", "średni", "wysoki"]
    return AssetRiskResponse(
        risk_category=labels[pred_raw],
        risk_prediction=probs[pred_raw]
    )


#Endpoint: AI Portfolio Risk
@router.get("/portfolio/{portfolio_id}/ai", response_model=PortfolioRiskResponse)
async def get_portfolio_risk_ai(
    portfolio_id: int,
    db: Session = Depends(get_db)
):
    """
    Prognoza ryzyka AI dla portfela (rolling window):
      1) Pobierz tickery z Transaction z net_qty > 0,
      2) Pobierz dane historyczne z Yahoo (5 lat) dla każdego tickera,
      3) Wylicz rolling window (20 dni) i cechy (return, abs_return, ma5, ma20, hv20, rsi14, macd, macd_signal),
      4) Wylicz wartość pozycji = net_qty * last_close,
      5) Zbuduj sekwencję portfela jako ważoną sumę sekwencji tickerów (ważone wartościami pozycji),
      6) Skaluj dane i predykuj klasę (0,1,2).
    """
    global risk_model, risk_scaler
    if risk_model is None or risk_scaler is None:
        raise HTTPException(status_code=503, detail="Model AI nie jest jeszcze dostępny")

    positions = get_current_portfolio_positions(db)
    if not positions:
        raise HTTPException(status_code=404, detail="Brak aktywnych pozycji na portfelu")

    feature_cols = ["return", "abs_return", "ma5", "ma20", "hv20", "rsi14", "macd", "macd_signal"]
    portfolio_sequence = np.zeros((WINDOW_SIZE, len(feature_cols)), dtype=np.float32)
    print(positions)
    total_values = []
    position_info = []

    for symbol, net_qty in positions:
        try:
            df = yf.download(symbol, period="5y", progress=False)
            if df.empty or len(df) < WINDOW_SIZE:
                continue

            tmp = df["Close"]
            returns = tmp.pct_change().fillna(0).squeeze()
            abs_return = returns.abs()

            ma5 = tmp.rolling(window=5).mean().pct_change().fillna(0)
            if isinstance(ma5, pd.DataFrame):
                ma5 = ma5.iloc[:, 0]
            ma5 = ma5.squeeze()

            ma20 = tmp.rolling(window=20).mean().pct_change().fillna(0)
            if isinstance(ma20, pd.DataFrame):
                ma20 = ma20.iloc[:, 0]
            ma20 = ma20.squeeze()

            hv20 = returns.rolling(window=20).std().fillna(0)
            if isinstance(hv20, pd.DataFrame):
                hv20 = hv20.iloc[:, 0]
            hv20 = hv20.squeeze()

            rsi14 = compute_rsi(tmp, period=14).fillna(0)
            if isinstance(rsi14, pd.DataFrame):
                rsi14 = rsi14.iloc[:, 0]
            rsi14 = rsi14.squeeze()

            macd, macd_signal = compute_macd(tmp)
            macd = macd.fillna(0)
            macd_signal = macd_signal.fillna(0)
            if isinstance(macd, pd.DataFrame):
                macd = macd.iloc[:, 0]
            macd = macd.squeeze()
            if isinstance(macd_signal, pd.DataFrame):
                macd_signal = macd_signal.iloc[:, 0]
            macd_signal = macd_signal.squeeze()

            df_feat = pd.DataFrame({
                "return": returns,
                "abs_return": abs_return,
                "ma5": ma5,
                "ma20": ma20,
                "hv20": hv20,
                "rsi14": rsi14,
                "macd": macd,
                "macd_signal": macd_signal
            })
            
            last_window = df_feat.iloc[-WINDOW_SIZE:].to_numpy(dtype=np.float32)
            
            last_close = float(tmp.dropna().iloc[-1])
            position_value = net_qty * last_close

            portfolio_sequence += position_value * last_window
            total_values.append(position_value)

        except Exception as e:
            print(e)

    print(total_values)
    if not total_values or np.all(portfolio_sequence == 0):
        raise HTTPException(status_code=400, detail=f"Brak danych historycznych do analizy portfela {portfolio_id}.")

    portfolio_sequence /= sum(total_values)

    num_features = portfolio_sequence.shape[1]
    scaled = risk_scaler.transform(portfolio_sequence)
    scaled = scaled.reshape(1, WINDOW_SIZE, num_features)

    probs = risk_model.predict(scaled).flatten().tolist()
    pred_raw = int(np.argmax(probs))
    labels = ["niski", "średni", "wysoki"]

    return PortfolioRiskResponse(
        portfolio_id=portfolio_id,
        risk_category=labels[pred_raw],
        risk_prediction=probs[pred_raw]
    )

