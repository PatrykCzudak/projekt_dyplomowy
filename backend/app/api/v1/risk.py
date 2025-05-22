from fastapi import APIRouter, HTTPException
from app.services import risk_analysis as ra

router = APIRouter(prefix="/risk", tags=["risk"])

@router.get("/asset/{symbol}/classical")
def asset_risk_classical(symbol: str, alpha: float = 0.05):
    try:
        prices = ra.get_historical_prices(symbol)
        returns = ra.compute_returns(prices)
        return {
            "symbol": symbol,
            "VaR_parametric": ra.var_parametric(returns, alpha),
            "VaR_historical": ra.var_historical(returns, alpha),
            "Expected_Shortfall": ra.expected_shortfall(returns, alpha),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/asset/{symbol}/ai")
def asset_risk_ai(symbol: str):
    try:
        prices = ra.get_historical_prices(symbol)
        returns = ra.compute_returns(prices)
        return {
            "symbol": symbol,
            "risk_category": ra.classify_risk(returns),
            "risk_prediction": ra.predict_risk(returns),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/portfolio/{user_id}/classical")
def portfolio_risk_classical(user_id: int, alpha: float = 0.05):
    try:
        returns = ra.get_portfolio_returns(user_id)
        return {
            "user_id": user_id,
            "VaR_parametric": ra.var_parametric(returns, alpha),
            "VaR_historical": ra.var_historical(returns, alpha),
            "Expected_Shortfall": ra.expected_shortfall(returns, alpha),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/portfolio/{user_id}/ai")
def portfolio_risk_ai(user_id: int):
    try:
        returns = ra.get_portfolio_returns(user_id)
        return {
            "user_id": user_id,
            "risk_category": ra.classify_risk(returns),
            "risk_prediction": ra.predict_risk(returns),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
