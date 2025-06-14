from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models import models
from app.schemas.asset import AssetCreate, AssetResponse
import yfinance as yf

router = APIRouter()


@router.get("/assets", response_model=list[AssetResponse])
def read_assets(db: Session = Depends(get_db)):
    return db.query(models.Asset).all()


@router.post("/assets", response_model=AssetResponse)
def create_asset(asset: AssetCreate, db: Session = Depends(get_db)):
    existing = db.query(models.Asset).filter(models.Asset.symbol == asset.symbol).first()
    if existing:
        raise HTTPException(status_code=400, detail="Asset already exists")

    new_asset = models.Asset(symbol=asset.symbol.upper(), name=asset.name)
    db.add(new_asset)
    db.commit()
    db.refresh(new_asset)
    return new_asset

@router.get("/assets/{symbol}/history")
def get_asset_history(symbol: str, period: str = "1mo"):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        if hist.empty:
            raise ValueError("Brak danych")

        prices = {
            str(date.date()): float(price)
            for date, price in hist["Close"].items()
        }

        data = []
        for date, row in hist.iterrows():
            data.append({
                "x": date.strftime('%Y-%m-%d'),
                "y": [  # OHLC
                    round(row["Open"], 2),
                    round(row["High"], 2),
                    round(row["Low"], 2),
                    round(row["Close"], 2)
                ]
            })
        
        return {
            "symbol": symbol.upper(),
            "historical_prices": prices,
            "ohlc": data
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Nie znaleziono danych: {e}")