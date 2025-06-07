from fastapi import APIRouter, HTTPException
from app.services import risk_analysis as ra

router = APIRouter(prefix="/risk", tags=["risk"])

# Endpointy do zwykłego ryzkyka 
@router.get("/asset/{symbol}/classical")
def asset_risk_classical(symbol: str, alpha: float = 0.05):
    try:
        prices = ra.get_historical_prices(symbol)
        returns = ra.compute_returns(prices)
        VaR_parametric = ra.var_parametric(returns, alpha)
        VaR_historical = ra.var_historical(returns, alpha)
        ES = ra.expected_shortfall(returns, alpha)

        returns_list = returns.tolist()
        dates_list = returns.index.strftime('%Y-%m-%d').tolist()

        return {
            "symbol": symbol,
            "returns": returns_list,
            "dates": dates_list,
            "VaR_parametric": VaR_parametric,
            "VaR_historical": VaR_historical,
            "Expected_Shortfall": ES,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
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


@router.get("/portfolio/{portfolio_id}/classical")
def portfolio_risk_classical(portfolio_id: int, alpha: float = 0.05):
    try:
        returns = ra.get_portfolio_returns(portfolio_id)
        VaR_parametric = ra.var_parametric(returns, alpha)
        VaR_historical = ra.var_historical(returns, alpha)
        ES = ra.expected_shortfall(returns, alpha)

        #P&L do histogramów
        returns_list = returns.tolist()
        dates_list = returns.index.strftime('%Y-%m-%d').tolist()

        return {
            "portfolio_id": portfolio_id,
            "returns": returns_list,
            "dates": dates_list,
            "VaR_parametric": VaR_parametric,
            "VaR_historical": VaR_historical,
            "Expected_Shortfall": ES,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
