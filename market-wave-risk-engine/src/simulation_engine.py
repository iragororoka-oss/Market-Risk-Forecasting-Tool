import numpy as np
import pandas as pd


class MarketWaveRiskEngine:
    """
    Monte Carlo market-risk engine for simulating future price paths
    and measuring boundary risk, Value-at-Risk, and Expected Shortfall.
    """

    def __init__(self, prices: pd.Series):
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        self.prices = prices.dropna().astype(float)
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        self.mu_daily = float(self.returns.mean())
        self.sigma_daily = float(self.returns.std())

    def simulate_paths(self, horizon_days=252, n_sims=10000, seed=42):
        np.random.seed(seed)

        s0 = self.prices.iloc[-1]
        dt = 1

        shocks = np.random.normal(
            0,
            1,
            size=(horizon_days, n_sims)
        )

        log_returns = (
            (self.mu_daily - 0.5 * self.sigma_daily ** 2) * dt
            + self.sigma_daily * np.sqrt(dt) * shocks
        )

        price_paths = np.zeros((horizon_days + 1, n_sims))
        price_paths[0] = s0

        for t in range(1, horizon_days + 1):
            price_paths[t] = price_paths[t - 1] * np.exp(log_returns[t - 1])

        return price_paths

    def boundary_probabilities(self, paths, upper_boundary, lower_boundary):
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
            "prob_hit_profit_first": hit_upper_first / total,
            "prob_hit_loss_first": hit_lower_first / total,
            "prob_no_boundary_hit": no_hit / total
        }

    def tail_risk(self, paths, confidence=0.95):
        s0 = paths[0, 0]
        final_prices = paths[-1]
        returns = (final_prices - s0) / s0

        losses = -returns

        var = np.quantile(losses, confidence)
        expected_shortfall = losses[losses >= var].mean()

        return {
            "VaR": var,
            "Expected Shortfall": expected_shortfall
        }