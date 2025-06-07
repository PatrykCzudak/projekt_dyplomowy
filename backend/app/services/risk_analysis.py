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
    Computes daily percentage returns from price data.
    """
    return prices.pct_change().dropna()

def var_parametric(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculates the parametric Value at Risk (VaR) assuming normal distribution.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = norm.ppf(alpha)
    return -(mu + sigma * z)

def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculates the historical Value at Risk (VaR) using percentile method.
    """
    return -np.percentile(returns, 100 * alpha)

def expected_shortfall(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calculates the Expected Shortfall (Conditional VaR) at the given alpha level.
    """
    threshold = np.percentile(returns, 100 * alpha)
    tail_losses = returns[returns <= threshold]
    return -tail_losses.mean()

def get_historical_prices(symbol: str, period: str = '5y') -> pd.Series:
    """
    Retrieves historical closing prices for a given symbol from Yahoo Finance.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError(f"No historical data for {symbol}")
    return hist["Close"]

def get_historical_data(symbol: str, period: str = '5y') -> pd.DataFrame:
    """
    Retrieves historical OHLCV (Open, High, Low, Close, Volume) data for a given symbol.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError(f"No historical data for {symbol}")
    hist = hist.reset_index()
    hist['Symbol'] = symbol
    return hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']]

#zwroty portfela
def get_portfolio_returns(_ignore, period: str = '5y') -> pd.Series:
    """
    Calculates the portfolio returns based on current net positions.
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
    rows = stmt.all()
    db.close()

    if not rows:
        raise ValueError("No active positions in the portfolio")

    symbols = [row.symbol for row in rows]
    net_map = {row.symbol: float(row.net_qty) for row in rows}

    data = yf.download(symbols, period=period)["Close"].dropna()
    if data.empty:
        raise ValueError("No historical data for assets in portfolio")

    returns = data.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough data to calculate portfolio returns.")

    last_prices = data.iloc[-1]
    weights = np.array([last_prices[s] * net_map[s] for s in symbols], dtype=float)
    total_value = weights.sum()
    if total_value == 0:
        raise ValueError("The total portfolio value is 0.")
    weights = weights / total_value

    portfolio_returns = returns.dot(weights)
    return portfolio_returns
