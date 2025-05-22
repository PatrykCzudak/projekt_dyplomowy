from pydantic import BaseModel

class PositionResponse(BaseModel):
    symbol: str
    name: str
    total_quantity: float
    average_price: float
    current_price: float
    change: float
