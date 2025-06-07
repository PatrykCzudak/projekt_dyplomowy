from pydantic import BaseModel, Field
from typing import Dict, List

class MarkowitzRequest(BaseModel):
    """
    Request model for Markowitz portfolio optimization.
    """
    gamma: float = Field(
        1.0,
        ge=0,
        description="Risk aversion coefficient (gamma). The higher it is, the greater the penalty for variance."
    )
    period: str = Field(
        '5y',
        description="Historical period (e.g. '1y', '5y')."
    )

class WeightsResponse(BaseModel):
    """
    Response model containing optimized portfolio weights and expected returns.
    """
    weights: Dict[str, float] = Field(
        ..., description="Mapping of asset ticker symbols to optimized weights"
    )
    mu: Dict[str, float] = Field(
        ..., description="Rates of return."
    )
        
class FrontierPoint(BaseModel):
    """
    Model representing a point on the efficient frontier.
    """
    gamma: float = Field(
        ..., description="Risk aversion parameter"
    )
    risk: float = Field(
        ..., description="Portfolio standard deviation"
    )
    expected_return: float = Field(
        ..., description="Portfolio expected return"
    )
