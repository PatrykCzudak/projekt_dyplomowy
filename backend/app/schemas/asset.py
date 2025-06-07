from pydantic import BaseModel, Field
from typing import Optional

class AssetCreate(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    name: Optional[str] = Field(None, description="Optional name of the asset.")

class AssetResponse(BaseModel):
    id: int = Field(..., description="Unique identifier of the asset.")
    symbol: str = Field(..., description="Ticker symbol of the asset.")
    name: Optional[str] = Field(None, description="Optional name of the asset.")

    class Config:
        from_attributes = True