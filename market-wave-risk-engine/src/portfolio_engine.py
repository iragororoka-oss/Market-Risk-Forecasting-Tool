import numpy as np
import pandas as pd


class PortfolioRiskEngine:
    """
    Portfolio-level Monte Carlo risk engine using correlated asset returns.
    """

    def __init__(self, prices: pd.DataFrame, weights):
        if not isinstance(prices, pd.DataFrame):
            raise ValueError("prices must be a pandas DataFrame with one column per asset.")

        self.prices = prices.dropna().astype(float)

        self.weights = np.array(weights, dtype=float)

        if len(self.weights) != self.prices.shape[1]:
            raise ValueError("Number of weights must match number of assets.")

        if self.weights.sum() == 0:
            raise ValueError("Weights cannot sum to zero.")

        self.weights = self.weights / self.weights.sum()

        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        self.mu_daily = self.returns.mean().values
        self.cov_daily = self.returns.cov().values
        self.corr_matrix = self.returns.corr()

        self.asset_names = list(self.prices.columns)

    def simulate_portfolio_paths(
        self,
        horizon_days: int = 60,
        n_sims: int = 5000,
        seed: int = 42,
        initial_value: float = 100.0
    ):
        """
        Simulates portfolio value paths using correlated asset returns.
        """

        rng = np.random.default_rng(seed)

        simulated_log_returns = rng.multivariate_normal(
            mean=self.mu_daily,
            cov=self.cov_daily,
            size=(horizon_days, n_sims)
        )

        simulated_simple_returns = np.exp(simulated_log_returns) - 1

        portfolio_returns = np.tensordot(
            simulated_simple_returns,
            self.weights,
            axes=([2], [0])
        )

        portfolio_paths = np.zeros((horizon_days + 1, n_sims))
        portfolio_paths[0] = initial_value

        for t in range(1, horizon_days + 1):
            portfolio_paths[t] = portfolio_paths[t - 1] * (1 + portfolio_returns[t - 1])

        return portfolio_paths

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
        initial_value = paths[0, 0]
        final_values = paths[-1]

        returns = (final_values - initial_value) / initial_value
        losses = -returns

        var = np.quantile(losses, confidence)
        expected_shortfall = losses[losses >= var].mean()

        return {
            "VaR": var,
            "Expected Shortfall": expected_shortfall
        }

    def risk_contribution(self):
        """
        Estimates each asset's contribution to total portfolio variance.
        """

        portfolio_variance = self.weights.T @ self.cov_daily @ self.weights

        marginal_contribution = self.cov_daily @ self.weights

        variance_contribution = self.weights * marginal_contribution

        percent_contribution = variance_contribution / portfolio_variance

        return pd.DataFrame({
            "Asset": self.asset_names,
            "Weight": self.weights,
            "Risk Contribution": percent_contribution
        })