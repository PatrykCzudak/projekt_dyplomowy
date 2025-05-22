from pydantic import BaseModel
from enum import Enum
from typing import Optional
from datetime import datetime

class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class TransactionCreate(BaseModel):
    symbol: str
    quantity: float
    price: float
    type: TransactionType

class TransactionResponse(BaseModel):
    id: int
    symbol: str
    name: Optional[str] = None
    quantity: float
    price: float
    type: TransactionType
    timestamp: Optional[datetime] = None
    
    class Config:
        from_attributes = True
