from pydantic import BaseModel, Field

class PositionResponse(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    name: str = Field(..., description="Name of the asset.")
    total_quantity: float = Field(..., description="Total quantity held in the portfolio.")
    average_price: float = Field(..., description="Average purchase price of the asset.")
    current_price: float = Field(..., description="Latest market price of the asset.")
    change: float = Field(..., description="Percentage change from the previous price.")

class AssetPosition(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    net_qty: float = Field(..., description="Net quantity held (buys minus sells).")

    class Config:
        orm_mode = True