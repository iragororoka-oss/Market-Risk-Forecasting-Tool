import numpy as np
import pandas as pd

from src.stress_testing import simulate_paths_with_params


def detect_volatility_regime(
    returns: pd.Series,
    lookback: int = 30
):
    """
    Detects the current volatility regime using rolling volatility.

    Regimes:
    - Calm: current rolling volatility is in the lower third of historical rolling volatility.
    - Normal: current rolling volatility is in the middle third.
    - High Volatility: current rolling volatility is in the upper third.
    """

    returns = returns.dropna().astype(float)

    rolling_vol = returns.rolling(lookback).std().dropna()

    if len(rolling_vol) < 60:
        raise ValueError("Not enough data to detect volatility regime.")

    current_vol = float(rolling_vol.iloc[-1])

    low_threshold = float(np.quantile(rolling_vol, 0.33))
    high_threshold = float(np.quantile(rolling_vol, 0.66))

    if current_vol <= low_threshold:
        regime = "Calm"
        sigma_multiplier = 0.85
        mu_shift = 0.0
    elif current_vol >= high_threshold:
        regime = "High Volatility"
        sigma_multiplier = 1.50
        mu_shift = -0.0003
    else:
        regime = "Normal"
        sigma_multiplier = 1.00
        mu_shift = 0.0

    rolling_table = pd.DataFrame({
        "Rolling Volatility": rolling_vol
    })

    summary = {
        "Current Regime": regime,
        "Current Rolling Volatility": current_vol,
        "Low Volatility Threshold": low_threshold,
        "High Volatility Threshold": high_threshold,
        "Sigma Multiplier": sigma_multiplier,
        "Drift Shift": mu_shift
    }

    return summary, rolling_table


def regime_adjusted_risk(
    engine,
    horizon_days: int,
    n_sims: int,
    upper_boundary: float,
    lower_boundary: float,
    lookback: int = 30,
    seed: int = 123
):
    """
    Runs a Monte Carlo simulation adjusted for the current volatility regime.
    """

    regime_summary, rolling_table = detect_volatility_regime(
        engine.returns,
        lookback=lookback
    )

    s0 = float(engine.prices.iloc[-1])

    adjusted_mu = float(engine.mu_daily + regime_summary["Drift Shift"])
    adjusted_sigma = float(engine.sigma_daily * regime_summary["Sigma Multiplier"])

    paths = simulate_paths_with_params(
        s0=s0,
        mu_daily=adjusted_mu,
        sigma_daily=adjusted_sigma,
        horizon_days=horizon_days,
        n_sims=n_sims,
        seed=seed
    )

    boundary_results = engine.boundary_probabilities(
        paths,
        upper_boundary,
        lower_boundary
    )

    risk_95 = engine.tail_risk(paths, confidence=0.95)
    risk_99 = engine.tail_risk(paths, confidence=0.99)

    result = {
        "regime_summary": regime_summary,
        "rolling_table": rolling_table,
        "paths": paths,
        "adjusted_mu": adjusted_mu,
        "adjusted_sigma": adjusted_sigma,
        "boundary_results": boundary_results,
        "risk_95": risk_95,
        "risk_99": risk_99
    }

    return result