"""This is a pipeline used to download date'a from yahoo finance.

---------
**With parameter `limit` in function `pipeline()`** - you can limit the number of processed symbols.
If `limit` is set to `None`, the pipeline processes all symbols.
Functions:
- `fetch_symbols_from_api(source)`: dowloads stock symbols from a specified source (e.g., S&P 500, NASDAQ 100).
- `load_seen()`: loads already processed symbols from a JSON file.
- `save_seen(symbols)`: saves processed symbols to a JSON file.
- 'quick_test(source, n, period)': downloads first *n* symbols from the specified source and runs the pipeline.
- 'pipeline(symbols, period, delay, limit=None)': processes the date'a for the given symbols.
- 'merge_all_csv(out_file)': merges all CSV files in the data directory into a single dataset.

Example usage:
>>> quick_test()                     
>>> syms = fetch_symbols_from_api("nasdaq100")
>>> pipeline(syms, period="5y", limit=10) 
"""

from __future__ import annotations

import json
import random
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Set

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SEEN_FILE = Path("symbols_seen.json")

# ────────────────────────────────────────────────────────────────────────────────
# SYMBOLS – pobieranie listy z zewnętrznego API / URL
# ────────────────────────────────────────────────────────────────────────────────

def fetch_symbols_from_api(source: str = "sp500") -> List[str]:
    """Returns a list of stock symbols from a specified source.
    Parametr `source` describe the source of symbols to fetch.
    Possible sources:
    - "sp500"(Wikipedia)
    - "nasdaq100"(Wikipedia)
    """

    if source == "sp500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        print(f"[INFO] I download S&P500 from {url} …")
        table = pd.read_html(url, match="Symbol")[0]
        return table["Symbol"].tolist()

    if source == "nasdaq100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        print(f"[INFO] I download S&P500 from {url} …")
        table = pd.read_html(url, match="Ticker")[0]
        return table["Ticker"].str.replace("\u200b", "").tolist()

    raise ValueError(f"Unavaible source: {source}")

# ────────────────────────────────────────────────────────────────────────────────
# UTIL: set już‑pobranych symboli
# ────────────────────────────────────────────────────────────────────────────────

def load_seen() -> Set[str]:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()

def save_seen(symbols: Set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(symbols), indent=2))

# ────────────────────────────────────────────────────────────────────────────────
# POBIERANIE I ZAPIS DANYCH
# ────────────────────────────────────────────────────────────────────────────────

def fetch_and_save(symbol: str, period: str = "max") -> pd.DataFrame | None:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        print(f"[WARN] No date'a for {symbol}")
        return None

    df = df.reset_index()
    df["Symbol"] = symbol.upper()
    out_path = DATA_DIR / f"{symbol.upper()}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved as {symbol} → {out_path} ({len(df)} rows)")
    return df

# ────────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ────────────────────────────────────────────────────────────────────────────────

def pipeline(
    symbols: Iterable[str],
    period: str = "max",
    delay: tuple[float, float] = (0.5, 1.5),
    limit: int | None = None,
) -> None:
    """
    Process date'a.
    Parameters:
    - symbols: str iterable
    - period: ex.("1y", "5y", "max")
    - delay: (min, max) sec between requests
    - limit: int | None
        if None, process all symbols;
        if int, process only that many *new* symbols
    """

    seen = load_seen()
    processed = 0

    for raw_symbol in symbols:
        if limit is not None and processed >= limit:
            print(f"[INFO] We reach a limit {limit} of companies, stopping pipeline.")
            break

        symbol = raw_symbol.upper().strip()
        if symbol in seen:
            print(f"[SKIP] {symbol} already processed, skipping.")
            continue

        try:
            df = fetch_and_save(symbol, period=period)
            if df is not None:
                seen.add(symbol)
                save_seen(seen)
                processed += 1
        except Exception as exc:
            print(f"[ERROR] {symbol}: {exc}")

        time.sleep(random.uniform(*delay))

# ────────────────────────────────────────────────────────────────────────────────
# QUICK TEST – tylko pierwsze N symboli
# ────────────────────────────────────────────────────────────────────────────────

def quick_test(source: str = "sp500", n: int = 10, period: str = "1y") -> None:
    """Downloads first N symbols from the specified source and runs the pipeline."""
    symbols = fetch_symbols_from_api(source)[:n]
    pipeline(symbols, period=period, limit=n)

# ────────────────────────────────────────────────────────────────────────────────
# MERGE
# ────────────────────────────────────────────────────────────────────────────────

def merge_all_csv(out_file: str | Path = "dataset.csv") -> None:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("[WARN] No CSV files found in the data directory.")
        return

    merged = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    merged.to_csv(out_file, index=False)
    print(f"[OK] Merged {len(csv_files)} files → {out_file} ({len(merged)} rows)")

# ────────────────────────────────────────────────────────────────────────────────
# CLI – Przykłady użycia
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Przykład 1: pełne S&P 500
    # symbols = fetch_symbols_from_api("sp500")
    # pipeline(symbols, period="5y")

    # Przykład 2: szybki test na 10 spółkach
    quick_test(source="sp500", n=2, period="1y")

    merge_all_csv()