import yfinance as yf

from src.portfolio_engine import PortfolioRiskEngine


tickers = ["AAPL", "MSFT", "NVDA"]
weights = [0.4, 0.35, 0.25]

data = yf.download(
    tickers,
    period="2y",
    auto_adjust=True
)

prices = data["Close"].dropna()

engine = PortfolioRiskEngine(prices, weights)

paths = engine.simulate_portfolio_paths(
    horizon_days=60,
    n_sims=5000,
    seed=10,
    initial_value=100
)

upper_boundary = 110
lower_boundary = 90

boundary_results = engine.boundary_probabilities(
    paths,
    upper_boundary,
    lower_boundary
)

risk_95 = engine.tail_risk(paths, confidence=0.95)
risk_99 = engine.tail_risk(paths, confidence=0.99)

print("\nPortfolio Assets")
print(tickers)

print("\nPortfolio Weights")
print(weights)

print("\nCorrelation Matrix")
print(engine.corr_matrix)

print("\nBoundary Results")
print(f"Probability of hitting profit target first: {boundary_results['prob_hit_profit_first']:.2%}")
print(f"Probability of hitting loss limit first: {boundary_results['prob_hit_loss_first']:.2%}")
print(f"Probability of hitting neither boundary: {boundary_results['prob_no_boundary_hit']:.2%}")

print("\nTail Risk")
print(f"95% VaR: {risk_95['VaR']:.2%}")
print(f"95% Expected Shortfall: {risk_95['Expected Shortfall']:.2%}")
print(f"99% VaR: {risk_99['VaR']:.2%}")
print(f"99% Expected Shortfall: {risk_99['Expected Shortfall']:.2%}")

print("\nRisk Contribution")
print(engine.risk_contribution())