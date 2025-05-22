from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.models import models
from app.schemas.portfolio import PositionResponse
from app.services.yahoo import get_current_price

router = APIRouter(
    prefix="/portfolio",
    tags=["portfolio"],
    include_in_schema=True
)

@router.get("/", response_model=List[PositionResponse])
def get_portfolio(db: Session = Depends(get_db)):
    transactions = db.query(models.Transaction).all()
    portfolio = {}

    for tx in transactions:
        asset = db.query(models.Asset).get(tx.asset_id)
        if not asset:
            continue
        symbol = asset.symbol

        if symbol not in portfolio:
            portfolio[symbol] = {
                "asset": asset,
                "quantity": 0.0,
                "cost": 0.0,
            }

        pos = portfolio[symbol]

        if tx.type == models.TransactionType.BUY:
            pos["quantity"] += tx.quantity
            pos["cost"] += tx.quantity * tx.price
        elif tx.type == models.TransactionType.SELL:
            if tx.quantity > pos["quantity"]:
                continue
            avg_price = pos["cost"] / pos["quantity"] if pos["quantity"] > 0 else 0
            pos["quantity"] -= tx.quantity
            pos["cost"] -= tx.quantity * avg_price

    response = []
    for symbol, pos in portfolio.items():
        if pos["quantity"] <= 0:
            continue

        avg_price = pos["cost"] / pos["quantity"]
        try:
            current_price, change = get_current_price(symbol)
        except Exception:
            current_price, change = 0.0, 0.0

        response.append(PositionResponse(
            symbol=symbol,
            name=pos["asset"].name,
            total_quantity=pos["quantity"],
            average_price=avg_price,
            current_price=current_price,
            change=change
        ))
    return response
