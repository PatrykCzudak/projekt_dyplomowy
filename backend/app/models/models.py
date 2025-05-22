from sqlalchemy import Column, Integer, Float, String, ForeignKey, Enum, DateTime, func
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.database import Base
import enum

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)

    transactions = relationship("Transaction", back_populates="asset")


class TransactionType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    type = Column(Enum(TransactionType), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    asset = relationship("Asset", back_populates="transactions")
    

