from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.models import models
from app.schemas.transaction import TransactionCreate, TransactionResponse
from typing import List
from fastapi.responses import JSONResponse
import yfinance as yf

router = APIRouter(
    prefix="/transactions",
    tags=["transactions"],
    include_in_schema=True
)

@router.post("/", response_model=TransactionResponse)
def create_transaction(tx: TransactionCreate, db: Session = Depends(get_db)):
    asset = db.query(models.Asset).filter(models.Asset.symbol == tx.symbol.upper()).first()
    if not asset:
        yf_data = yf.Ticker(tx.symbol.upper()).info
        name = yf_data.get("shortName") or tx.symbol.upper()

        asset = models.Asset(
            symbol=tx.symbol.upper(),
            name=name
        )
        db.add(asset)
        db.commit()
        db.refresh(asset)

    transaction = models.Transaction(
        asset_id=asset.id,
        quantity=tx.quantity,
        price=tx.price,
        type=tx.type
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    return TransactionResponse(
        id=transaction.id,
        symbol=asset.symbol,
        name=asset.name if asset else None,
        quantity=transaction.quantity,
        price=transaction.price,
        type=transaction.type,
        timestamp=transaction.timestamp
    )

@router.get("/", response_model=List[TransactionResponse])
def list_transactions(db: Session = Depends(get_db)):
    transactions = db.query(models.Transaction).all()
    result = []
    for t in transactions:
        asset = db.query(models.Asset).get(t.asset_id)
        result.append(TransactionResponse(
            id=t.id,
            symbol=asset.symbol if asset else "UNKNOWN",
            name=asset.name if asset else None,
            quantity=t.quantity,
            price=t.price,
            type=t.type,
            timestamp=t.timestamp
        ))
    return result

@router.options("/")
def options_transactions():
    return JSONResponse(content={"detail": "ok"})
