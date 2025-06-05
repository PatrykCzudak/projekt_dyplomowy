import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.db.database import SessionLocal
from app.models import models
from sqlalchemy import case, func

#Klasyczne metody dla pojedynczego instrumentu
def compute_returns(prices: pd.Series) -> pd.Series:
    """
    Oblicza zwroty procentowe (Close_t / Close_{t-1} - 1) i usuwa NaN.
    """
    return prices.pct_change().dropna()

def var_parametric(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Parametryczny VaR (zakłada rozkład normalny).
    """
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(alpha)
    return -(mu + sigma * z)

def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historyczny VaR (percentyl).
    """
    return -np.percentile(returns, 100 * alpha)

def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Expected Shortfall (średnia zwrotów poniżej percentyla alpha).
    """
    threshold = np.percentile(returns, 100 * alpha)
    tail_losses = returns[returns <= threshold]
    return -tail_losses.mean()

def get_historical_prices(symbol: str, period: str = '5y') -> pd.Series:
    """
    Pobiera z Yahoo Finance historyczne ceny CH_close (domyślny okres = '1y').
    Zwraca Series zamknięć lub rzuca ValueError, jeśli brak danych.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError(f"Brak danych historycznych dla {symbol}")
    return hist["Close"]

def get_historical_data(symbol: str, period: str = '5y') -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError(f"Brak danych historycznych dla {symbol}")
    hist = hist.reset_index()
    hist['Symbol'] = symbol
    return hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]

#zwroty portfela
def get_portfolio_returns(_ignore, period: str = '5y') -> pd.Series:
    """
    Oblicza zwroty portfela na podstawie WSZYSTKICH transakcji w tabeli:
      1) Jedno zapytanie:
         SELECT Asset.symbol, SUM(CASE WHEN type='BUY' THEN quantity ELSE -quantity END) AS net_qty
         FROM assets JOIN transactions ON assets.id=transactions.asset_id
         GROUP BY Asset.symbol
         HAVING net_qty > 0
      2) Pobiera z Yahoo Finance historyczne ceny Close dla uzyskanej listy symboli (parametr `period`).
      3) Oblicza dzienne zwroty procentowe (returns).
      4) Ostatnie ceny (last_prices) i wagi = (last_price * net_qty) / suma(last_price * net_qty).
      5) Zwraca Series zwrotów portfela = returns.dot(weights).
    """
    db = SessionLocal()

    qty_case = case(
        (models.Transaction.type == "BUY",  models.Transaction.quantity),
        (models.Transaction.type == "SELL", -models.Transaction.quantity),
        else_=0.0
    )
    stmt = (
        db.query(
            models.Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(models.Transaction, models.Asset.id == models.Transaction.asset_id)
        .group_by(models.Asset.symbol)
        .having(func.sum(qty_case) > 0)
    )
    rows = stmt.all()  # [(symbol1, net1), (symbol2, net2), …]
    db.close()

    if not rows:
        raise ValueError("Brak aktywnych pozycji w portfelu")

    symbols = [row.symbol for row in rows]
    net_map = {row.symbol: float(row.net_qty) for row in rows}

    data = yf.download(symbols, period=period)["Close"].dropna()
    if data.empty:
        raise ValueError("Brak danych historycznych dla aktywów w portfelu")

    returns = data.pct_change().dropna()
    if returns.empty:
        raise ValueError("Za mało danych, aby obliczyć zwroty portfela")

    last_prices = data.iloc[-1]
    weights = np.array([last_prices[s] * net_map[s] for s in symbols], dtype=float)
    total_value = weights.sum()
    if total_value == 0:
        raise ValueError("Całkowita wartość portfela wynosi 0")
    weights = weights / total_value

    portfolio_returns = returns.dot(weights)
    return portfolio_returns
