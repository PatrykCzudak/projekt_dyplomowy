import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from cvxpy import Variable, quad_form, Problem, Maximize, OSQP
from sqlalchemy.orm import Session
from sqlalchemy import case, func
from app.models import models
from app.services.risk_analysis import get_historical_data

BASE_DIR = os.path.join(os.path.dirname(__file__), '../../../AI')

def load_models():
    models_dict = {}

    # Model 1: Prediction Model
    pred_model_path = os.path.join(BASE_DIR, 'prediction_model.h5')
    scaler_pred_path = os.path.join(BASE_DIR, 'prediction_model_scaler.pkl')
    models_dict['prediction'] = tf.keras.models.load_model(pred_model_path, compile=False)
    with open(scaler_pred_path, 'rb') as f:
        models_dict['prediction_scaler'] = pickle.load(f)

    # Model 2: Risk LSTM
    risk_model_path = os.path.join(BASE_DIR, 'risk_lstm_model.h5')
    scaler_risk_path = os.path.join(BASE_DIR, 'scaler_lstm.pkl')
    models_dict['risk'] = tf.keras.models.load_model(risk_model_path, compile=False)
    with open(scaler_risk_path, 'rb') as f:
        models_dict['risk_scaler'] = pickle.load(f)

    # Model 3: Recommendation Model
    rec_model_path = os.path.join(BASE_DIR, 'recommendation_model.pkl')
    scaler_rec_path = os.path.join(BASE_DIR, 'recommendation_model_scaler.pkl')
    with open(rec_model_path, 'rb') as f:
        models_dict['recommendation'] = pickle.load(f)
    with open(scaler_rec_path, 'rb') as f:
        models_dict['recommendation_scaler'] = pickle.load(f)

    print("AI models loaded successfully.")
    return models_dict

MODELS = load_models()

def ai_markowitz_optimize(db: Session, gamma: float, period: str = '5y', top_n: int = 5) -> dict:

    qty_case = case(
        (models.Transaction.type == "BUY", models.Transaction.quantity),
        (models.Transaction.type == "SELL", -models.Transaction.quantity),
        else_=0.0
    )
    rows = (
        db.query(
            models.Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(models.Transaction, models.Asset.id == models.Transaction.asset_id)
        .group_by(models.Asset.id, models.Asset.symbol)
        .having(func.sum(qty_case) > 0)
        .all()
    )

    symbols = [row.symbol for row in rows]
    if not symbols:
        raise ValueError("No active positions in the portfolio.")

    dfs = []
    for sym in symbols:
        prices = get_historical_data(sym, period=period)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame().reset_index()
        prices['Symbol'] = sym
        dfs.append(prices)
    df = pd.concat(dfs)

    recommended_symbols = recommend_assets(df, top_n=top_n)
    if not recommended_symbols:
        raise ValueError("No recommended symbols found.")

    mu_dict = predict_mu(df, recommended_symbols)
    if not mu_dict:
        raise ValueError("Failed to predict expected returns.")

    returns_list = []
    for sym in recommended_symbols:
        sub_df = df[df['Symbol'] == sym].sort_values(['Date'])
        sub_df['Return'] = sub_df['Close'].pct_change()
        returns_list.append(sub_df['Return'].dropna().values)

    returns_df = pd.DataFrame(returns_list).T
    returns_df.columns = recommended_symbols
    Sigma = returns_df.cov().values

    mu = np.array([mu_dict[sym] for sym in recommended_symbols])
    n = len(mu)
    w = Variable(n)

    obj = Maximize(mu.T @ w - (gamma/2) * quad_form(w, Sigma))
    constraints = [w >= 0.01, sum(w) == 1]
    prob = Problem(obj, constraints)
    prob.solve(solver=OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver status: {prob.status}")

    weights = {sym: float(w_val) for sym, w_val in zip(recommended_symbols, w.value)}
    return {
        "weights": weights,
        "mu": mu_dict
    }

def recommend_assets(df: pd.DataFrame, top_n: int = 5):
    scaler = MODELS['recommendation_scaler']
    model = MODELS['recommendation']

    df = df.sort_values(['Symbol', 'Date'])
    df['Return'] = df.groupby('Symbol')['Close'].pct_change()
    df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x / x.shift(20) - 1)
    df = df.dropna(subset=['Return', 'Momentum'])

    features = ['Return', 'Momentum']
    recommended = []

    for symbol in df['Symbol'].unique():
        sub_df = df[df['Symbol'] == symbol].iloc[-1:]
        X = sub_df[features].values
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        recommended.append((symbol, pred))

    recommended = sorted(recommended, key=lambda x: x[1], reverse=True)
    top_symbols = [sym for sym, _ in recommended[:top_n]]
    return top_symbols

def predict_mu(df: pd.DataFrame, symbols: list):
    scaler = MODELS['prediction_scaler']
    model = MODELS['prediction']
    SEQ_LEN = 20

    mu_dict = {}
    for sym in symbols:
        sub_df = df[df['Symbol'] == sym].sort_values(['Date']).copy()
        sub_df['Return'] = sub_df['Close'].pct_change()
        sub_df['RSI'] = sub_df['Close'].transform(compute_rsi)
        sub_df['MACD'] = sub_df['Close'].transform(lambda x: compute_macd(x)[0])
        sub_df['MACD_signal'] = sub_df['Close'].transform(lambda x: compute_macd(x)[1])
        sub_df = sub_df.dropna(subset=['Return', 'RSI', 'MACD', 'MACD_signal'])

        features = ['Close', 'Volume', 'Return', 'RSI', 'MACD', 'MACD_signal']
        if len(sub_df) < SEQ_LEN:
            continue

        try:
            for col in features:
                sub_df[f'Scaled_{col}'] = scaler.transform(sub_df[[col]])
        except Exception as e:
            continue

        seq = sub_df.iloc[-SEQ_LEN:][[f'Scaled_{col}' for col in features]].values
        seq = np.expand_dims(seq, axis=0)
        pred = model.predict(seq)[0][0]
        mu_dict[sym] = pred

    return mu_dict

def compute_rsi(series, period=14):
    series = series.sort_index()
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    gain = up.rolling(window=period).mean()
    loss = down.rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    series = series.sort_index()
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal
