"""Pipeline do pobierania danych z Yahoo Finance i budowania zbioru danych.

---------
1. **Parametr `limit` w funkcji `pipeline()`** możesz zatrzymać pobieranie np. na 10 pierwszych spółkach.
2. **Skrót `quick_test()`** – wywołaj jedną funkcją, aby pobrać *tylko* pierwsze *N* symboli (domyślnie 10) z wybranego źródła.

Funkcje:
- `fetch_symbols_from_api(source)`: pobiera listę tickerów (S&P 500 lub Nasdaq 100)
- `quick_test(source="sp500", n=10, period="1y")`: pobiera *n* symboli i przekazuje do `pipeline()`
- `pipeline(symbols, period, delay, limit=None)`: wykonuje pełny proces pobierania; gdy `limit` ≠ None, zatrzymuje się po tylu spółkach
- `merge_all_csv()`: scala pojedyncze pliki w jeden dataset

Przykłady:
>>> quick_test()                     # 10 pierwszych z S&P 500
>>> syms = fetch_symbols_from_api("nasdaq100")
>>> pipeline(syms, period="5y", limit=10)  # również 10 pierwszych
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
    """Zwróć listę tickerów z wybranego źródła.

    Obsługiwane źródła:
    - "sp500": tabela S&P 500 (Wikipedía)
    - "nasdaq100": tabela Nasdaq‑100 (Wikipedía)
    """

    if source == "sp500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        print(f"[INFO] Pobieram listę S&P 500 z {url} …")
        table = pd.read_html(url, match="Symbol")[0]
        return table["Symbol"].tolist()

    if source == "nasdaq100":
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        print(f"[INFO] Pobieram listę Nasdaq‑100 z {url} …")
        table = pd.read_html(url, match="Ticker")[0]
        return table["Ticker"].str.replace("\u200b", "").tolist()

    raise ValueError(f"Nieobsługiwane źródło: {source}")

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
        print(f"[WARN] Brak danych dla {symbol}")
        return None

    df = df.reset_index()
    df["Symbol"] = symbol.upper()
    out_path = DATA_DIR / f"{symbol.upper()}.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Zapisano {symbol} → {out_path} ({len(df)} wierszy)")
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
    """Przetwórz listę symboli.

    Parametry:
    - symbols: str iterable
    - period: okres do pobrania (np. "1y", "5y", "max")
    - delay: (min, max) sekund między zapytaniami
    - limit: jeżeli podany, pipeline zakończy się po przetworzeniu tylu *nowych* spółek
    """

    seen = load_seen()
    processed = 0

    for raw_symbol in symbols:
        if limit is not None and processed >= limit:
            print(f"[INFO] Osiągnięto limit {limit} spółek – zatrzymuję pipeline")
            break

        symbol = raw_symbol.upper().strip()
        if symbol in seen:
            print(f"[SKIP] {symbol} już pobrany, pomijam")
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
    """Pobierz n pierwszych symboli z podanego źródła i uruchom pipeline()."""
    symbols = fetch_symbols_from_api(source)[:n]
    pipeline(symbols, period=period, limit=n)

# ────────────────────────────────────────────────────────────────────────────────
# MERGE
# ────────────────────────────────────────────────────────────────────────────────

def merge_all_csv(out_file: str | Path = "dataset.csv") -> None:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("[WARN] Brak plików CSV do połączenia")
        return

    merged = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    merged.to_csv(out_file, index=False)
    print(f"[OK] Scalono {len(csv_files)} plików → {out_file} ({len(merged)} wierszy)")

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