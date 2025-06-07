from pydantic import BaseModel, Field

class AssetRiskResponse(BaseModel):
    risk_category: str = Field(..., description="Risk category.")
    risk_prediction: float = Field(..., description="Predicted risk score.")

class PortfolioRiskResponse(BaseModel):
    portfolio_id: int = Field(..., description="Unique identifier of the portfolio.")
    risk_category: str = Field(..., description="Risk category.")
    risk_prediction: float = Field(..., description="Predicted risk score.")