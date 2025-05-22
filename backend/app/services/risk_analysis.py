import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from app.db.database import SessionLocal
from app.models import models

# --- Klasyczne metody ---
def compute_returns(prices):
    return prices.pct_change().dropna()

def var_parametric(returns, alpha=0.05):
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(alpha)
    return -(mu + sigma * z)

def var_historical(returns, alpha=0.05):
    return -np.percentile(returns, 100 * alpha)

def expected_shortfall(returns, alpha=0.05):
    threshold = np.percentile(returns, 100 * alpha)
    tail_losses = returns[returns <= threshold]
    return -tail_losses.mean()

# --- AI ---
clf = DecisionTreeClassifier().fit([[0.01], [0.05], [0.15]], ['Low', 'Medium', 'High'])
reg = DecisionTreeRegressor().fit([[0.01], [0.05], [0.15]], [0.012, 0.055, 0.152])

def classify_risk(returns):
    sigma = returns.std()
    return clf.predict([[sigma]])[0]

def predict_risk(returns):
    sigma = returns.std()
    return reg.predict([[sigma]])[0]

# --- Dane ---
def get_historical_prices(symbol, period='1y'):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError("Brak danych")
    return hist["Close"]

def get_portfolio_returns(user_id, period='1y'):
    db = SessionLocal()
    txs = db.query(models.Transaction).filter(models.Transaction.user_id == user_id).all()
    if not txs:
        raise ValueError("Brak transakcji")

    symbols = list({t.asset.symbol for t in txs})
    shares_map = {s: 0 for s in symbols}
    for t in txs:
        shares_map[t.asset.symbol] += t.quantity if t.type == "BUY" else -t.quantity

    data = yf.download(symbols, period=period)["Close"].dropna()
    returns = data.pct_change().dropna()
    last_prices = data.iloc[-1]

    weights = np.array([last_prices[s] * shares_map[s] for s in symbols])
    total = weights.sum()
    if total == 0:
        raise ValueError("Brak wartości portfela")

    weights = weights / total
    return returns.dot(weights)
