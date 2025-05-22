from pydantic import BaseModel
from typing import Optional


class AssetCreate(BaseModel):
    symbol: str
    name: Optional[str] = None


class AssetResponse(BaseModel):
    id: int
    symbol: str
    name: Optional[str]

    class Config:
        from_attributes = True
