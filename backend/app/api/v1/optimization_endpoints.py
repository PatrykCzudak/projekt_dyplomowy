from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from typing import List
from app.db.database import get_db
from app.services.optimization import markowitz_optimize, efficient_frontier, portfolio_cloud
from app.schemas.optimization_schema import WeightsResponse, MarkowitzRequest, FrontierPoint
from app.services.markowitz_ai import ai_markowitz_optimize

router = APIRouter(
    prefix="/optimize",
    tags=["optimization"]
)

@router.post(
    "/markowitz",
    response_model=WeightsResponse,
    summary="Portfolio optymalization Markowitz",
    description="Return weights for portfolio otimization."
)
async def optimize_markowitz(
    req: MarkowitzRequest,
    db: Session = Depends(get_db)
):
    try:
        weights = await run_in_threadpool(markowitz_optimize, db, req.gamma, req.period)
        return WeightsResponse(weights=weights, mu={})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {e}")

#AIAJ -------------------------
@router.post(
    "/markowitz-ai",
    response_model=WeightsResponse,
    summary="AI-Enhanced Markowitz Optimization",
    description="Performs Markowitz optimization using AI-enhanced μ and recommendations."
)
async def optimize_markowitz_ai(
    gamma: float = Query(10.0, ge=0),
    period: str = Query('5y'),
    top_n: int = Query(20, ge=1, le=20),
    db: Session = Depends(get_db)
):
    try:
        result = await run_in_threadpool(ai_markowitz_optimize, db, gamma, period, top_n)
        return WeightsResponse(weights=result['weights'], mu=result['mu'])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

#--------------------------------------- Frontierek 
@router.get(
    "/frontier",
    response_model=List[FrontierPoint],
    summary="Efficient frontier",
    description="Returns Effective Frontier points."
)
async def get_frontier(
    gamma_min: float = Query(0.0, ge=0),
    gamma_max: float = Query(10.0, ge=0),
    num_points: int = Query(50, ge=2, le=200),
    period: str = Query('5y', description="Historical data period (e.g. '1y', '5y')."),
    db: Session = Depends(get_db)
):
    try:
        pts = await run_in_threadpool(
            efficient_frontier, db, num_points, gamma_min, gamma_max, period
        )
        return pts
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Server error while calculating efective frontier: {e}")
    
@router.get(
    "/cloud",
    response_model=List[FrontierPoint],
    summary="Portfolio cloud",
    description="Returns random portfolios (risk and return) for visualization."
)
async def get_portfolio_cloud(
    num_points: int = Query(5000, ge=100, le=20000),
    period: str = Query('5y', description="Historical data period (e.g. '1y', '5y')."),
    db: Session = Depends(get_db)
):
    try:
        pts = await run_in_threadpool(portfolio_cloud, db, num_points, period)
        return pts
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Server error while generating cloud: {e}")
