from pydantic import BaseModel, Field
from typing import Dict, List

class MarkowitzRequest(BaseModel):
    gamma: float = Field(
        1.0,
        ge=0,
        description="Współczynnik awersji do ryzyka (gamma). Im wyższy – większa kara za wariancję."
    )
    period: str = Field(
        '5y',
        description="Okres historyczny danych do analizy (np. '1y', '5y')."
    )

class WeightsResponse(BaseModel):
    """
    Response model for portfolio optimization endpoints.

    Attributes:
        weights (Dict[str, float]):
            Mapping of asset tickers to their optimized portfolio weights.
    """
    weights: Dict[str, float] = Field(
        ..., description="Mapping of asset ticker symbols to optimized weights"
    )

    class Config:
        schema_extra = {
            "example": {
                "weights": {
                    "AAPL": 0.30,
                    "MSFT": 0.25,
                    "GOOG": 0.45
                }
            }
        }
        
class FrontierPoint(BaseModel):
    gamma: float = Field(..., description="Risk aversion parameter")
    risk: float = Field(..., description="Portfolio standard deviation")
    expected_return: float = Field(..., description="Portfolio expected return")
