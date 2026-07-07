import numpy as np
import pandas as pd


def historical_var_es(
    returns: pd.Series,
    confidence: float = 0.95
):
    """
    Calculates Historical Simulation Value-at-Risk and Expected Shortfall.

    Instead of assuming normally distributed returns, this method uses
    the empirical historical return distribution directly.
    """

    returns = returns.dropna().astype(float)

    if len(returns) < 30:
        raise ValueError("Not enough return data for historical simulation.")

    losses = -returns

    var = np.quantile(losses, confidence)

    tail_losses = losses[losses >= var]

    expected_shortfall = tail_losses.mean()

    return {
        "Historical VaR": var,
        "Historical Expected Shortfall": expected_shortfall,
        "Worst Historical Loss": losses.max(),
        "Average Historical Loss": losses.mean()
    }


def historical_boundary_probability(
    returns: pd.Series,
    current_price: float,
    upper_boundary: float,
    lower_boundary: float,
    horizon_days: int = 60,
    n_sims: int = 5000,
    seed: int = 42
):
    """
    Simulates future paths by randomly resampling historical returns.

    This is a historical simulation approach instead of a normal Monte Carlo model.
    """

    returns = returns.dropna().astype(float).values

    if len(returns) < 30:
        raise ValueError("Not enough return data for historical boundary simulation.")

    rng = np.random.default_rng(seed)

    sampled_returns = rng.choice(
        returns,
        size=(horizon_days, n_sims),
        replace=True
    )

    paths = np.zeros((horizon_days + 1, n_sims))
    paths[0] = current_price

    for t in range(1, horizon_days + 1):
        paths[t] = paths[t - 1] * np.exp(sampled_returns[t - 1])

    hit_upper_first = 0
    hit_lower_first = 0
    no_hit = 0

    for i in range(paths.shape[1]):
        path = paths[:, i]

        upper_hits = np.where(path >= upper_boundary)[0]
        lower_hits = np.where(path <= lower_boundary)[0]

        if len(upper_hits) == 0 and len(lower_hits) == 0:
            no_hit += 1
        elif len(upper_hits) == 0:
            hit_lower_first += 1
        elif len(lower_hits) == 0:
            hit_upper_first += 1
        elif upper_hits[0] < lower_hits[0]:
            hit_upper_first += 1
        else:
            hit_lower_first += 1

    total = paths.shape[1]

    return {
        "paths": paths,
        "prob_hit_profit_first": hit_upper_first / total,
        "prob_hit_loss_first": hit_lower_first / total,
        "prob_no_boundary_hit": no_hit / total
    }