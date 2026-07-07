import numpy as np
import pandas as pd


def simulate_paths_with_params(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    horizon_days: int = 60,
    n_sims: int = 5000,
    seed: int = 7,
    initial_shock: float = 0.0
):
    """
    Simulates price paths using custom drift, volatility, and optional initial shock.
    """

    rng = np.random.default_rng(seed)

    shocks = rng.normal(
        0,
        1,
        size=(horizon_days, n_sims)
    )

    log_returns = (
        (mu_daily - 0.5 * sigma_daily ** 2)
        + sigma_daily * shocks
    )

    # Apply an immediate market shock in the first simulated step.
    if initial_shock != 0:
        log_returns[0, :] += np.log1p(initial_shock)

    price_paths = np.zeros((horizon_days + 1, n_sims))
    price_paths[0] = s0

    for t in range(1, horizon_days + 1):
        price_paths[t] = price_paths[t - 1] * np.exp(log_returns[t - 1])

    return price_paths


def run_stress_tests(
    engine,
    horizon_days: int,
    n_sims: int,
    upper_boundary: float,
    lower_boundary: float,
    seed: int = 7
) -> pd.DataFrame:
    """
    Runs multiple stress scenarios and returns a comparison table.
    """

    s0 = float(engine.prices.iloc[-1])

    scenarios = {
        "Base Case": {
            "mu_shift": 0.0,
            "sigma_multiplier": 1.0,
            "initial_shock": 0.0
        },
        "High Volatility": {
            "mu_shift": 0.0,
            "sigma_multiplier": 2.0,
            "initial_shock": 0.0
        },
        "Bearish Drift": {
            "mu_shift": -0.001,
            "sigma_multiplier": 1.25,
            "initial_shock": 0.0
        },
        "Market Shock": {
            "mu_shift": -0.0005,
            "sigma_multiplier": 1.5,
            "initial_shock": -0.08
        },
        "Recovery Scenario": {
            "mu_shift": 0.0007,
            "sigma_multiplier": 1.2,
            "initial_shock": -0.05
        }
    }

    results = []

    for scenario_name, config in scenarios.items():
        scenario_mu = float(engine.mu_daily + config["mu_shift"])
        scenario_sigma = float(engine.sigma_daily * config["sigma_multiplier"])

        paths = simulate_paths_with_params(
            s0=s0,
            mu_daily=scenario_mu,
            sigma_daily=scenario_sigma,
            horizon_days=horizon_days,
            n_sims=n_sims,
            seed=seed,
            initial_shock=config["initial_shock"]
        )

        boundary_results = engine.boundary_probabilities(
            paths,
            upper_boundary,
            lower_boundary
        )

        risk_95 = engine.tail_risk(paths, confidence=0.95)
        risk_99 = engine.tail_risk(paths, confidence=0.99)

        results.append({
            "Scenario": scenario_name,
            "Daily Drift": scenario_mu,
            "Daily Volatility": scenario_sigma,
            "Initial Shock": config["initial_shock"],
            "Profit First": boundary_results["prob_hit_profit_first"],
            "Loss First": boundary_results["prob_hit_loss_first"],
            "Neither Boundary": boundary_results["prob_no_boundary_hit"],
            "95% VaR": risk_95["VaR"],
            "95% Expected Shortfall": risk_95["Expected Shortfall"],
            "99% VaR": risk_99["VaR"],
            "99% Expected Shortfall": risk_99["Expected Shortfall"]
        })

    return pd.DataFrame(results)