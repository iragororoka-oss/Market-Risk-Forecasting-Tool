import numpy as np
import pandas as pd

from src.simulation_engine import MarketWaveRiskEngine
from src.portfolio_engine import PortfolioRiskEngine
from src.stress_testing import run_stress_tests
from src.historical_simulation import historical_var_es, historical_boundary_probability
from src.backtesting import run_var_backtest


def make_fake_prices(n_days=400, start_price=100):
    np.random.seed(1)
    returns = np.random.normal(0.0005, 0.01, n_days)
    prices = start_price * np.exp(np.cumsum(returns))
    return pd.Series(prices)


def make_fake_portfolio_prices(n_days=400):
    np.random.seed(2)

    returns = np.random.normal(
        loc=[0.0004, 0.0005, 0.0003],
        scale=[0.01, 0.012, 0.009],
        size=(n_days, 3)
    )

    prices = 100 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(
        prices,
        columns=["AAPL", "MSFT", "NVDA"]
    )


def test_historical_var_es_outputs_valid_values():
    prices = make_fake_prices()
    engine = MarketWaveRiskEngine(prices)

    result = historical_var_es(engine.returns, confidence=0.95)

    assert "Historical VaR" in result
    assert "Historical Expected Shortfall" in result
    assert result["Historical Expected Shortfall"] >= result["Historical VaR"]


def test_historical_boundary_probabilities_sum_to_one():
    prices = make_fake_prices()
    engine = MarketWaveRiskEngine(prices)

    current_price = float(prices.iloc[-1])

    result = historical_boundary_probability(
        returns=engine.returns,
        current_price=current_price,
        upper_boundary=current_price * 1.10,
        lower_boundary=current_price * 0.90,
        horizon_days=30,
        n_sims=1000,
        seed=3
    )

    total = (
        result["prob_hit_profit_first"]
        + result["prob_hit_loss_first"]
        + result["prob_no_boundary_hit"]
    )

    assert abs(total - 1.0) < 1e-10


def test_stress_testing_returns_all_scenarios():
    prices = make_fake_prices()
    engine = MarketWaveRiskEngine(prices)

    current_price = float(prices.iloc[-1])

    results = run_stress_tests(
        engine=engine,
        horizon_days=30,
        n_sims=1000,
        upper_boundary=current_price * 1.10,
        lower_boundary=current_price * 0.90,
        seed=4
    )

    assert len(results) == 5
    assert "Base Case" in results["Scenario"].values
    assert "High Volatility" in results["Scenario"].values
    assert "Market Shock" in results["Scenario"].values


def test_portfolio_engine_simulates_paths():
    prices = make_fake_portfolio_prices()
    weights = [0.4, 0.35, 0.25]

    engine = PortfolioRiskEngine(prices, weights)

    paths = engine.simulate_portfolio_paths(
        horizon_days=30,
        n_sims=1000,
        seed=5,
        initial_value=10000
    )

    assert paths.shape == (31, 1000)
    assert paths[0, 0] == 10000


def test_portfolio_risk_contribution_sums_to_one():
    prices = make_fake_portfolio_prices()
    weights = [0.4, 0.35, 0.25]

    engine = PortfolioRiskEngine(prices, weights)

    contribution = engine.risk_contribution()

    total_contribution = contribution["Risk Contribution"].sum()

    assert abs(total_contribution - 1.0) < 1e-6


def test_var_backtest_outputs_summary_and_table():
    prices = make_fake_prices(n_days=500)
    engine = MarketWaveRiskEngine(prices)

    summary, table = run_var_backtest(
        returns=engine.returns,
        confidence=0.95,
        rolling_window=252,
        n_sims=1000,
        seed=6
    )

    assert "Observed Breach Rate" in summary
    assert "Expected Breach Rate" in summary
    assert len(table) > 0