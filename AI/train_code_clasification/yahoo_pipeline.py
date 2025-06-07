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


#lista tickerów
def fetch_symbols_from_api(source: str = "sp500") -> List[str]:
    """Returns a list of stock symbols from a specified source.
    Parametr `source` describe the source of symbols to fetch.
    Possible sources:
    - "sp500"(Wikipedia)
    - "nasdaq100"(Wikipedia)
    - "custom": lista tickerów o podwyższonej zmienności
    - "all": sp500 + nasdaq100 + custom
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

    if source == "custom":
        etf_sectorowe = [
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLB", "XLY", "XLU", "XBI", "KWEB",
            "XRT", "SMH", "XHB", "SOXX", "IBB", "XLRE", "XME", "XOP", "XTL", "XAR",
            "XSW", "XSD", "ITA", "XNTK", "XTN", "IYT", "IYZ", "XLFN", "XLFK",
            "VGT", "VHT", "VDE", "VFH", "VPU", "VOX", "VDC", "VCR", "VAW", "VIS",
            "VBR", "VBK", "VTV", "VONV", "VOE", "VOT", "VUG", "MGK", "MGV", "IWF",
            "IWD", "IWO", "IWN", "IWP", "IWS", "IWR", "IWB", "IWM", "IWV", "IWY",
            "IWL", "IWO", "IWN", "IWP", "IWS", "IWR", "IWB", "IWM", "IWV", "IWY",
            "IWL", "SPY", "DIA", "QQQ", "VTI", "VOO", "IVV", "SCHX", "SCHG", "SCHV",
            "SCHA", "SCHB", "SCHM", "SCHR", "SCHD", "SCHH", "SCHF", "SCHE", "SCHO",
            "SCHP", "SCHZ", "SPSM", "SPYG", "SPYV", "SPYB", "SPYD", "SPYV", "SPYG"
        ]

        globalne_indeksy = [
            "EEM", "EWZ", "FXI", "TUR", "RSX", "INDA", "VNM", "EWJ", "EWG", "EWC",
            "ASHR", "VEA", "VWO", "EWU", "EWT", "EWL", "EWQ", "EWY", "EZA", "EWW",
            "EWS", "EWN", "EWI", "EWK", "EWD", "EWP", "EPU", "SPY", "IVV", "VOO",
            "VTI", "VT", "ACWI", "URTH", "VEU", "VSS", "SCZ", "IEFA", "IEMG", "EMB",
            "BNDX", "AGG", "LQD", "HYG", "JNK", "TIP", "SHY", "IEI", "IEF", "TLT",
            "SPTL", "SPTI", "SPTS", "SPMB", "SPAB", "SPBO", "SPHY", "SPIP", "SPTI",
            "SPTL", "SPTS", "SPAB", "SPMB", "SPHY", "SPIP", "SPTI", "SPTL", "SPTS"
        ]

        spolki_wysoka_zmiennosc = [
            "PLTR", "TSLA", "NVDA", "AMD", "BYND", "COIN", "HOOD", "RIVN", "SOFI",
            "RBLX", "SPCE", "ROKU", "SQ", "AFRM", "UPST", "LCID", "NKLA", "SPOT",
            "ZM", "CRWD", "ZS", "NET", "SNOW", "AI", "NIO", "LI", "XPEV", "BIDU",
            "JD", "PDD", "BABA", "TME", "U", "RIOT", "MARA", "SHOP", "COUP", "ZI",
            "DOCU", "TEAM", "MDB", "DDOG", "ABNB", "PINS", "PATH", "OKTA", "FSLY",
            "TWLO", "SNAP", "CHWY", "BMBL", "DASH", "GME", "AMC", "BB", "NOK", "TLRY",
            "CGC", "CRON", "ACB", "SNDL", "APHA", "OGI", "HEXO", "VFF", "GRWG", "AAPL",
            "MSFT", "GOOGL", "AMZN", "META", "NFLX", "INTC", "CSCO", "ORCL", "IBM",
            "ADBE", "CRM", "PYPL", "EBAY", "QCOM", "TXN", "AVGO", "MU", "LRCX", "KLAC"
        ]

        surowce_volatility = [
            "GDX", "USO", "GLD", "SLV", "DBC", "VXX", "UVXY", "VIXY", "UNG", "BNO",
            "DBA", "UUP", "SCO", "BOIL", "KOLD", "UGAZ", "DGAZ", "WEAT", "SOYB",
            "NIB", "BAL", "CORN", "SGG", "JJG", "CANE", "PALL", "PPLT", "DBB", "DBE",
            "DBP", "DBO", "DBA", "DBS", "DBV", "DJP", "DYY", "DZZ", "GLTR", "IAU",
            "JJN", "JJP", "JJU", "JJC", "JJG", "JJM", "JJN", "JJT", "JJS", "JJU",
            "JJC", "JJG", "JJM", "JJN", "JJT", "JJS", "JJU", "JJC", "JJG", "JJM"
        ]
        
        inne_spolki = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "JPM", "JNJ",
            "V", "PG", "UNH", "HD", "MA", "NVDA", "DIS", "PYPL", "BAC", "VZ", "ADBE",
            "CMCSA", "NFLX", "INTC", "T", "KO", "PFE", "PEP", "CSCO", "XOM", "ABT",
            "CRM", "CVX", "NKE", "MRK", "WMT", "ORCL", "TMO", "MCD", "COST", "WFC",
            "MDT", "DHR", "ACN", "AVGO", "TXN", "NEE", "LLY", "PM", "UNP", "LIN",
            "HON", "UPS", "QCOM", "AMGN", "LOW", "IBM", "BA", "SBUX", "MMM", "RTX",
            "INTU", "GE", "CAT", "BLK", "AXP", "ISRG", "SPGI", "GILD", "LMT", "CVS",
            "BKNG", "ADI", "ZTS", "SYK", "MDLZ", "DE", "PLD", "CB", "CI", "USB",
            "MO", "TGT", "FIS", "DUK", "SO", "BDX", "PNC", "APD", "C", "ADP", "ICE",
            "SHW", "MMC", "TJX", "GM", "VRTX", "REGN", "EW", "HUM", "ITW", "FISV"
        ]

        tickers = (
            etf_sectorowe +
            globalne_indeksy +
            spolki_wysoka_zmiennosc +
            surowce_volatility +
            inne_spolki
        )

        tickers = (
            etf_sectorowe +
            globalne_indeksy +
            spolki_wysoka_zmiennosc +
            surowce_volatility
        )
        return tickers

    if source == "all":
        tickers = []
        tickers.extend(fetch_symbols_from_api("sp500"))
        tickers.extend(fetch_symbols_from_api("nasdaq100"))
        tickers.extend(fetch_symbols_from_api("custom"))
        print(f"[INFO] Pobieram {len(tickers)} tickerów z wszystkich źródeł")
        return tickers

    raise ValueError(f"Unavaible source: {source}")

def load_seen() -> Set[str]:
    if SEEN_FILE.exists():
        return set(json.loads(SEEN_FILE.read_text()))
    return set()

def save_seen(symbols: Set[str]) -> None:
    SEEN_FILE.write_text(json.dumps(sorted(symbols), indent=2))

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

def pipeline(
    symbols: Iterable[str],
    period: str = "max",
    delay: tuple[float, float] = (0.5, 1.5),
    limit: int | None = None,
) -> None:
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

def merge_all_csv(out_file: str | Path = "dataset.csv") -> None:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print("[WARN] No CSV files found in the data directory.")
        return

    merged = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
    merged.to_csv(out_file, index=False)
    print(f"[OK] Merged {len(csv_files)} files → {out_file} ({len(merged)} rows)")


if __name__ == "__main__":

    symbols = fetch_symbols_from_api("all")
    pipeline(symbols, period="5y")

    merge_all_csv()