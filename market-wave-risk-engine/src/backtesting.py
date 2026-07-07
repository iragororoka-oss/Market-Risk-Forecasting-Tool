import numpy as np
import pandas as pd


def run_var_backtest(
    returns: pd.Series,
    confidence: float = 0.95,
    rolling_window: int = 252,
    n_sims: int = 10000,
    seed: int = 42
):
    """
    Rolling one-day VaR backtest.

    For each day, the model estimates VaR using only prior returns.
    Then it checks whether the next realized loss exceeds the VaR estimate.
    """

    returns = returns.dropna().astype(float)

    if len(returns) <= rolling_window + 20:
        raise ValueError("Not enough data for backtesting. Try a longer historical period.")

    rng = np.random.default_rng(seed)

    results = []

    for i in range(rolling_window, len(returns)):
        train_returns = returns.iloc[i - rolling_window:i]
        realized_return = returns.iloc[i]

        mu = train_returns.mean()
        sigma = train_returns.std()

        simulated_returns = rng.normal(
            loc=mu,
            scale=sigma,
            size=n_sims
        )

        simulated_losses = -simulated_returns
        var_estimate = np.quantile(simulated_losses, confidence)

        realized_loss = -realized_return
        breach = realized_loss > var_estimate

        results.append({
            "Date": returns.index[i],
            "Realized Return": realized_return,
            "Realized Loss": realized_loss,
            "VaR Estimate": var_estimate,
            "Breach": breach
        })

    backtest_table = pd.DataFrame(results)

    observed_breaches = int(backtest_table["Breach"].sum())
    total_observations = len(backtest_table)

    expected_breach_rate = 1 - confidence
    observed_breach_rate = observed_breaches / total_observations

    summary = {
        "Confidence Level": confidence,
        "Total Observations": total_observations,
        "Expected Breach Rate": expected_breach_rate,
        "Observed Breach Rate": observed_breach_rate,
        "Expected Breaches": total_observations * expected_breach_rate,
        "Observed Breaches": observed_breaches,
        "Average VaR": backtest_table["VaR Estimate"].mean(),
        "Maximum Realized Loss": backtest_table["Realized Loss"].max()
    }

    return summary, backtest_table