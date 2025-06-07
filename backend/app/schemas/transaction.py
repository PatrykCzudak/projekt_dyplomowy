from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime

class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class TransactionCreate(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    quantity: float = Field(..., description="Quantity involved in the transaction.")
    price: float = Field(..., description="Transaction price per unit.")
    type: TransactionType = Field(..., description="Type of the transaction.")

class TransactionResponse(BaseModel):
    id: int = Field(..., description="Unique identifier of the transaction.")
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    name: Optional[str] = Field(None, description="Optional name of the asset.")
    quantity: float = Field(..., description="Quantity involved in the transaction.")
    price: float = Field(..., description="Transaction price per unit.")
    type: TransactionType = Field(..., description="Type of the transaction.")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of the transaction.")

    class Config:
        from_attributes = True