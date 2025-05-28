import numpy as np
import pandas as pd
from cvxpy import Variable, quad_form, Problem, Maximize, OSQP
from typing import Dict, List
from sqlalchemy.orm import Session
from app.services.risk_analysis import get_historical_prices, compute_returns
from app.models import models

def markowitz_optimize(db: Session, gamma: float) -> Dict[str, float]:
    symbols = [s[0] for s in db.query(models.Asset.symbol).all()]
    if not symbols:
        raise ValueError("Brak aktywów w bazie. Dodaj przynajmniej jeden Asset przez /assets")

    # zwroty
    returns_list = []
    for sym in symbols:
        prices = get_historical_prices(sym)
        ret = compute_returns(prices)
        ret.name = sym
        returns_list.append(ret)
    returns_df = pd.concat(returns_list, axis=1)

    # Parametry Markowitza
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)
    w = Variable(n)

    obj = Maximize(mu.T @ w - (gamma/2) * quad_form(w, Sigma))
    constraints = [w >= 0, sum(w) == 1]
    prob = Problem(obj, constraints)

    prob.solve(solver=OSQP)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Solver zwrócił status {prob.status}")

    return dict(zip(returns_df.columns.tolist(), w.value.tolist()))

# AI do stworzenia
def ai_optimize(db: Session) -> Dict[str, float]:
    symbols = [s[0] for s in db.query(models.Asset.symbol).all()]
    n = len(symbols)
    weight = 1.0 / n if n else 0.0
    return {sym: weight for sym in symbols}

def efficient_frontier(
    db: Session,
    num_points: int = 50,
    gamma_min: float = 0.0,
    gamma_max: float = 10.0
) -> List[Dict[str, float]]:
    symbols = [s[0] for s in db.query(models.Asset.symbol).all()]
    if not symbols:
        raise ValueError("Brak aktywów w bazie")

    returns_list = []
    for sym in symbols:
        prices = get_historical_prices(sym)
        r = compute_returns(prices)
        r.name = sym
        returns_list.append(r)
    returns_df = pd.concat(returns_list, axis=1)
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    n = len(mu)

    gammas = np.linspace(gamma_min, gamma_max, num_points)
    frontier = []

    for gamma in gammas:
        w = Variable(n)
        obj = Maximize(mu.T @ w - (gamma/2) * quad_form(w, Sigma))
        cons = [w >= 0, sum(w) == 1]
        prob = Problem(obj, cons)
        prob.solve(solver=OSQP, warm_start=True)
        if prob.status not in ("optimal","optimal_inaccurate"):
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