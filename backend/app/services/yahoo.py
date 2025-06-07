import yfinance as yf

def get_current_price(symbol: str) -> tuple[float, float]:
    """
    Retrieves the most recent closing price and its percentage change from the previous day.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2d")

    if 'Close' not in hist.columns or len(hist) == 0:
        raise ValueError(f"Not enough data for symbol {symbol}")

    current_price = hist['Close'].iloc[-1]

    if len(hist) >= 2:
        previous_price = hist['Close'].iloc[-2]
        change = ((current_price - previous_price) / previous_price) * 100
    else:
        change = 0.0
    return current_price, change
