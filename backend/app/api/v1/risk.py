from fastapi import APIRouter, HTTPException
from app.services import risk_analysis as ra

router = APIRouter(prefix="/risk", tags=["risk"])

# Endpointy do zwykłego ryzkyka 
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
    
@router.get("/asset/{symbol}/classical")
def asset_risk_classical(symbol: str, alpha: float = 0.05):
    """
    Klasyczna analiza ryzyka dla pojedynczego tickeru:
      - VaR parametryczny
      - VaR historyczny
      - Expected Shortfall (ES)
    """
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


@router.get("/portfolio/{portfolio_id}/classical")
def portfolio_risk_classical(portfolio_id: int, alpha: float = 0.05):
    """
    Klasyczna analiza ryzyka dla portfela (wszystkie aktywne pozycje w bazie):
      - VaR parametryczny
      - VaR historyczny
      - Expected Shortfall (ES)
    """
    try:
        returns = ra.get_portfolio_returns(portfolio_id)
        return {
            "portfolio_id": portfolio_id,
            "VaR_parametric": ra.var_parametric(returns, alpha),
            "VaR_historical": ra.var_historical(returns, alpha),
            "Expected_Shortfall": ra.expected_shortfall(returns, alpha),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
