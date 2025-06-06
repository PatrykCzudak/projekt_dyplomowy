import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from cvxpy import Variable, quad_form, Problem, Maximize, OSQP, sum_squares
from sqlalchemy.orm import Session
from sqlalchemy import case, func
from app.models import models
from app.services.risk_analysis import get_historical_data

BASE_DIR = os.path.join(os.path.dirname(__file__), '../../../AI')

def load_models():
    models_dict = {}

    # Model 1: Prediction Model (CNN+Transformer)
    pred_model_path = os.path.join(BASE_DIR, 'prediction_model_5d.h5')
    scaler_pred_path = os.path.join(BASE_DIR, 'prediction_model_5d_scaler.pkl')
    models_dict['prediction'] = tf.keras.models.load_model(pred_model_path, compile=False)
    with open(scaler_pred_path, 'rb') as f:
        models_dict['prediction_scaler'] = pickle.load(f)

    # Model 2: Recommendation Model (CatBoost)
    rec_model_path = os.path.join(BASE_DIR, 'recommendation_model.pkl')
    with open(rec_model_path, 'rb') as f:
        models_dict['recommendation'] = pickle.load(f)

    print("AI models loaded successfully.")
    return models_dict

MODELS = load_models()

def ai_markowitz_optimize(db: Session, gamma: float, period: str = '5y', top_n: int = 5, lambda_reg: float = 0.15) -> dict:
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

    # Rolling covariance (window=60)
    window_size = 60
    returns_df = pd.DataFrame()
    for sym in recommended_symbols:
        sub_df = df[df['Symbol'] == sym].sort_values(['Date'])
        sub_df['Return'] = sub_df['Close'].pct_change()
        returns_df[sym] = sub_df['Return'].reset_index(drop=True)

    returns_df = returns_df.dropna()
    rolling_returns = returns_df.tail(window_size)
    Sigma = rolling_returns.cov().values

    mu = np.array([mu_dict[sym] for sym in recommended_symbols])
    mu_shifted = mu - mu.min() + 0.01

    n = len(mu_shifted)
    w = Variable(n)


    obj = Maximize(mu_shifted.T @ w - (gamma/2) * quad_form(w, Sigma) - lambda_reg * sum_squares(w))
    constraints = [w >= 0.01, sum(w) == 1]
    prob = Problem(obj, constraints)
    prob.solve(solver=OSQP)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver status: {prob.status}")

    weights = {sym: float(w_val) for sym, w_val in zip(recommended_symbols, w.value)}
    return {
        "weights": weights,
        "mu": {sym: round(float(mu_val), 6) for sym, mu_val in zip(recommended_symbols, mu)}
    }

def recommend_assets(df: pd.DataFrame, top_n: int = 5):
    model = MODELS['recommendation']

    df = df.sort_values(['Symbol', 'Date'])
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

    df['Stochastic_14'] = df.groupby('Symbol')['Close'].transform(
        lambda x: 100 * (x - x.rolling(window=14).min()) / 
                (x.rolling(window=14).max() - x.rolling(window=14).min())
    )

    df['Volatility_20'] = df.groupby('Symbol')['Return'].transform(lambda x: x.rolling(window=20).std())

    df = df.dropna(subset=[
        'Return', 'Momentum', 'Return_lag1', 'Return_lag5',
        'RollingMean_5', 'RollingStd_5', 'MACD',
        'Bollinger_Width', 'Stochastic_14', 'Volatility_20'
    ])

    df['Symbol_Code'] = df['Symbol'].astype('category').cat.codes

    features = ['Return', 'Momentum', 'Return_lag1', 'Return_lag5', 
                'RollingMean_5', 'RollingStd_5', 'MACD', 
                'Bollinger_Width', 'Stochastic_14', 'Volatility_20', 'Symbol']

    recommended = []
    for symbol in df['Symbol'].unique():
        sub_df = df[df['Symbol'] == symbol].iloc[-1:]
        X = sub_df[features]
        pred = model.predict(X)[0]
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
        sub_df['Momentum'] = sub_df['Close'].pct_change(periods=5)
        sub_df['MA5'] = sub_df['Close'].rolling(window=5).mean()
        sub_df['STD5'] = sub_df['Close'].rolling(window=5).std()

        sub_df['Symbol_Code'] = sub_df['Symbol'].astype('category').cat.codes
        sub_df = sub_df.dropna(subset=[
            'Return', 'RSI', 'MACD', 'MACD_signal', 'Momentum', 'MA5', 'STD5'
        ])

        features = [
            'Close', 'Volume', 'Return', 'RSI', 'MACD', 'MACD_signal',
            'Momentum', 'MA5', 'STD5'
        ]

        if len(sub_df) < SEQ_LEN:
            continue

        scaled_features = scaler.transform(sub_df[features])
        sub_df_scaled = pd.DataFrame(
            scaled_features,
            columns=[f'Scaled_{col}' for col in features],
            index=sub_df.index
        )
        sub_df = pd.concat([sub_df, sub_df_scaled], axis=1)

        seq_other = sub_df.iloc[-SEQ_LEN:][[f'Scaled_{col}' for col in features]].values
        seq_symbol = sub_df.iloc[-SEQ_LEN:]['Symbol_Code'].values.reshape(-1, 1)
        seq_other = np.expand_dims(seq_other, axis=0)
        seq_symbol = np.expand_dims(seq_symbol, axis=0)

        pred = model.predict({'other_features': seq_other, 'symbol_code': seq_symbol})[0][0]
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
