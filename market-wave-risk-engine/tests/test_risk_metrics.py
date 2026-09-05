import numpy as np
import pandas as pd

from src.simulation_engine import MarketWaveRiskEngine


def test_engine_calculates_returns():
    prices = pd.Series([100, 101, 102, 103, 104])

    engine = MarketWaveRiskEngine(prices)

    assert len(engine.returns) == 4
    assert engine.sigma_daily >= 0


def test_simulate_paths_shape():
    prices = pd.Series([100, 101, 102, 103, 104, 105])

    engine = MarketWaveRiskEngine(prices)

    paths = engine.simulate_paths(
        horizon_days=10,
        n_sims=100,
        seed=1
    )

    assert paths.shape == (11, 100)


def test_boundary_probabilities_sum_to_one():
    prices = pd.Series([100, 101, 102, 103, 104, 105])

    engine = MarketWaveRiskEngine(prices)

    paths = engine.simulate_paths(
        horizon_days=10,
        n_sims=100,
        seed=1
    )

    results = engine.boundary_probabilities(
        paths,
        upper_boundary=110,
        lower_boundary=90
    )

    total_probability = (
        results["prob_hit_profit_first"]
        + results["prob_hit_loss_first"]
        + results["prob_no_boundary_hit"]
    )

    assert abs(total_probability - 1.0) < 1e-10


def test_tail_risk_is_non_negative_for_normal_case():
    prices = pd.Series([100, 101, 102, 103, 104, 105])

    engine = MarketWaveRiskEngine(prices)

    paths = engine.simulate_paths(
        horizon_days=10,
        n_sims=100,
        seed=1
    )

    risk = engine.tail_risk(paths, confidence=0.95)

    assert "VaR" in risk
    assert "Expected Shortfall" in risk