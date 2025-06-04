import numpy as np
import pandas as pd
from cvxpy import Variable, quad_form, Problem, Maximize, OSQP
from typing import Dict, List
from sqlalchemy.orm import Session
from app.services.risk_analysis import get_historical_prices, compute_returns
from app.models import models
from sqlalchemy import case, func

def markowitz_optimize(db: Session, gamma: float, period: str = '5y') -> Dict[str, float]:
    """
    Optymalizacja portfela metodą Markowitza:
    - Używa tylko aktywów, które mają net_qty > 0 (z bazy).
    - Pozwala określić okres historyczny (np. '1y', '5y').
    - Zawiera ograniczenie dywersyfikacji: min udział aktywa = 1%.
    """
    qty_case = case(
        (models.Transaction.type == "BUY",  models.Transaction.quantity),
        (models.Transaction.type == "SELL", -models.Transaction.quantity),
        else_=0.0
    )
    rows = (
        db.query(
            models.Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(models.Transaction, models.Asset.id == models.Transaction.asset_id)
        .group_by(models.Asset.id, models.Asset.symbol)
        .having(func.sum(qty_case) > 0)
        .all()
    )

    symbols = [row.symbol for row in rows]
    if not symbols:
        raise ValueError("Brak aktywnych pozycji w portfelu")

    returns_list = []
    for sym in symbols:
        prices = get_historical_prices(sym, period=period)
        ret = compute_returns(prices)
        ret.name = sym
        returns_list.append(ret)

    returns_df = pd.concat(returns_list, axis=1)

    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    print(f"[DEBUG] mu: {mu}")
    print(f"[DEBUG] Sigma:\n{Sigma}")

    n = len(mu)
    w = Variable(n)

    obj = Maximize(mu.T @ w - (gamma / 2) * quad_form(w, Sigma))
    constraints = [w >= 0.01, sum(w) == 1]
    prob = Problem(obj, constraints)

    prob.solve(solver=OSQP)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver zwrócił status {prob.status}")

    return dict(zip(returns_df.columns.tolist(), w.value.tolist()))

def portfolio_cloud(
    db: Session,
    num_points: int = 5000,
    period: str = '5y'
) -> List[Dict[str, float]]:
    """
    Generuje chmurę portfeli z losowymi wagami.
    """
    qty_case = case(
        (models.Transaction.type == "BUY", models.Transaction.quantity),
        (models.Transaction.type == "SELL", -models.Transaction.quantity),
        else_=0.0
    )
    rows = (
        db.query(
            models.Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(models.Transaction, models.Asset.id == models.Transaction.asset_id)
        .group_by(models.Asset.id, models.Asset.symbol)
        .having(func.sum(qty_case) > 0)
        .all()
    )

    symbols = [row.symbol for row in rows]
    if not symbols:
        raise ValueError("No active positions in portfolio")

    returns_list = []
    for sym in symbols:
        prices = get_historical_prices(sym, period=period)
        r = compute_returns(prices)
        r.name = sym
        returns_list.append(r)

    returns_df = pd.concat(returns_list, axis=1)
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values

    n = len(mu)
    points = []

    for _ in range(num_points):
        w = np.random.rand(n)
        w /= w.sum()  # normalize

        expected_return = float(mu.T @ w)
        risk = float(np.sqrt(w.T @ Sigma @ w))

        points.append({
            "gamma": 0,
            "risk": risk,
            "expected_return": expected_return
        })

    return points


def efficient_frontier(db: Session, num_points: int = 50, gamma_min: float = 0.0, gamma_max: float = 10.0, period: str = '5y') -> List[Dict[str, float]]:
    """
    Generuje punkty na efektywnej granicy ryzyka i zwrotu.
    """
    qty_case = case(
        (models.Transaction.type == "BUY",  models.Transaction.quantity),
        (models.Transaction.type == "SELL", -models.Transaction.quantity),
        else_=0.0
    )
    rows = (
        db.query(
            models.Asset.symbol.label("symbol"),
            func.sum(qty_case).label("net_qty")
        )
        .join(models.Transaction, models.Asset.id == models.Transaction.asset_id)
        .group_by(models.Asset.id, models.Asset.symbol)
        .having(func.sum(qty_case) > 0)
        .all()
    )

    symbols = [row.symbol for row in rows]
    if not symbols:
        raise ValueError("Brak aktywnych pozycji w portfelu")

    returns_list = []
    for sym in symbols:
        prices = get_historical_prices(sym, period=period)
        r = compute_returns(prices)
        r.name = sym
        returns_list.append(r)

    returns_df = pd.concat(returns_list, axis=1)
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    print(f"[DEBUG] mu: {mu}")
    print(f"[DEBUG] Sigma:\n{Sigma}")

    n = len(mu)
    gammas = np.linspace(gamma_min, gamma_max, num_points)
    frontier = []

    for gamma in gammas:
        w = Variable(n)
        obj = Maximize(mu.T @ w - (gamma / 2) * quad_form(w, Sigma))
        constraints = [w >= 0.01, sum(w) == 1]
        prob = Problem(obj, constraints)
        prob.solve(solver=OSQP, warm_start=True)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            continue
        wv = np.array(w.value).flatten()
        port_ret = float(mu.T @ wv)
        port_risk = float(np.sqrt(wv.T @ Sigma @ wv))
        frontier.append({
            "gamma": float(gamma),
            "risk": port_risk,
            "expected_return": port_ret
        })
    return frontier

# AI do stworzenia
def ai_optimize(db: Session) -> Dict[str, float]:
    symbols = [s[0] for s in db.query(models.Asset.symbol).all()]
    n = len(symbols)
    weight = 1.0 / n if n else 0.0
    return {sym: weight for sym in symbols}
