import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from src.portfolio_engine import PortfolioRiskEngine


st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide"
)

st.title("Portfolio Risk Dashboard")

st.write(
    """
    This page extends the market-risk engine from a single asset to a portfolio of assets.
    It uses correlated Monte Carlo simulation to estimate portfolio-level boundary risk,
    Value-at-Risk, Expected Shortfall, and asset risk contributions.
    """
)


# --------------------------------------------------
# Sidebar inputs
# --------------------------------------------------

st.sidebar.header("Portfolio Inputs")

tickers_input = st.sidebar.text_input(
    "Tickers separated by commas",
    value="AAPL, MSFT, NVDA"
)

weights_input = st.sidebar.text_input(
    "Weights separated by commas",
    value="0.4, 0.35, 0.25"
)

period = st.sidebar.selectbox(
    "Historical period",
    ["1y", "2y", "5y"],
    index=1
)

horizon_days = st.sidebar.slider(
    "Simulation horizon",
    min_value=10,
    max_value=252,
    value=60
)

n_sims = st.sidebar.slider(
    "Number of simulations",
    min_value=1000,
    max_value=20000,
    value=5000,
    step=1000
)

profit_target = st.sidebar.slider(
    "Profit target",
    min_value=1,
    max_value=50,
    value=10
) / 100

loss_limit = st.sidebar.slider(
    "Loss limit",
    min_value=1,
    max_value=50,
    value=10
) / 100
initial_value = st.sidebar.number_input(
    "Portfolio starting value",
    min_value=100.0,
    value=10000.0,
    step=1000.0
)

# --------------------------------------------------
# Parse inputs
# --------------------------------------------------

tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

try:
    weights = [float(w.strip()) for w in weights_input.split(",")]
except ValueError:
    st.error("Weights must be numbers separated by commas.")
    st.stop()

if len(tickers) != len(weights):
    st.error("The number of tickers must match the number of weights.")
    st.stop()

if sum(weights) == 0:
    st.error("Weights cannot sum to zero.")
    st.stop()


# --------------------------------------------------
# Download data
# --------------------------------------------------

data = yf.download(
    tickers,
    period=period,
    auto_adjust=True,
    progress=False
)

if data.empty:
    st.error("No market data found. Try different tickers.")
    st.stop()

prices = data["Close"].dropna()

if isinstance(prices, pd.Series):
    prices = prices.to_frame(name=tickers[0])

prices = prices.dropna().astype(float)


# --------------------------------------------------
# Run portfolio engine
# --------------------------------------------------

try:
    engine = PortfolioRiskEngine(prices, weights)
except ValueError as error:
    st.error(str(error))
    st.stop()

paths = engine.simulate_portfolio_paths(
    horizon_days=horizon_days,
    n_sims=n_sims,
    seed=10,
    initial_value=initial_value
)

upper_boundary = initial_value * (1 + profit_target)
lower_boundary = initial_value * (1 - loss_limit)

boundary_results = engine.boundary_probabilities(
    paths,
    upper_boundary,
    lower_boundary
)

risk_95 = engine.tail_risk(paths, confidence=0.95)
risk_99 = engine.tail_risk(paths, confidence=0.99)

risk_contribution = engine.risk_contribution()


# --------------------------------------------------
# Display metrics
# --------------------------------------------------

st.subheader("Portfolio Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Portfolio Starting Value", f"${initial_value:,.2f}")
col2.metric("Profit Boundary", f"${upper_boundary:.2f}")
col3.metric("Loss Boundary", f"${lower_boundary:.2f}")

col4, col5, col6 = st.columns(3)

col4.metric(
    "Hit Profit Target First",
    f"{boundary_results['prob_hit_profit_first']:.2%}"
)

col5.metric(
    "Hit Loss Limit First",
    f"{boundary_results['prob_hit_loss_first']:.2%}"
)

col6.metric(
    "Hit Neither Boundary",
    f"{boundary_results['prob_no_boundary_hit']:.2%}"
)

col7, col8, col9, col10 = st.columns(4)

col7.metric("95% VaR", f"{risk_95['VaR']:.2%}")
col8.metric("95% Expected Shortfall", f"{risk_95['Expected Shortfall']:.2%}")
col9.metric("99% VaR", f"{risk_99['VaR']:.2%}")
col10.metric("99% Expected Shortfall", f"{risk_99['Expected Shortfall']:.2%}")


# --------------------------------------------------
# Portfolio paths plot
# --------------------------------------------------

st.subheader("Simulated Portfolio Value Paths")

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(min(100, paths.shape[1])):
    ax.plot(paths[:, i], linewidth=0.8, alpha=0.5)

ax.axhline(upper_boundary, linestyle="--", label="Profit Boundary")
ax.axhline(lower_boundary, linestyle="--", label="Loss Boundary")

ax.set_title("Correlated Monte Carlo Portfolio Simulation")
ax.set_xlabel("Days")
ax.set_ylabel("Portfolio Value")
ax.legend()

st.pyplot(fig)


# --------------------------------------------------
# Correlation matrix
# --------------------------------------------------

st.subheader("Asset Correlation Matrix")

st.dataframe(
    engine.corr_matrix,
    use_container_width=True
)


# --------------------------------------------------
# Risk contribution
# --------------------------------------------------

st.subheader("Portfolio Risk Contribution")

risk_display = risk_contribution.copy()
risk_display["Weight"] = risk_display["Weight"].map(lambda x: f"{x:.2%}")
risk_display["Risk Contribution"] = risk_display["Risk Contribution"].map(lambda x: f"{x:.2%}")

st.dataframe(
    risk_display,
    use_container_width=True
)


# --------------------------------------------------
# Download results
# --------------------------------------------------

portfolio_results = pd.DataFrame([
    {
        "Tickers": ", ".join(tickers),
        "Weights": ", ".join([str(w) for w in weights]),
        "Profit First": boundary_results["prob_hit_profit_first"],
        "Loss First": boundary_results["prob_hit_loss_first"],
        "Neither Boundary": boundary_results["prob_no_boundary_hit"],
        "95% VaR": risk_95["VaR"],
        "95% Expected Shortfall": risk_95["Expected Shortfall"],
        "99% VaR": risk_99["VaR"],
        "99% Expected Shortfall": risk_99["Expected Shortfall"]
    }
])

st.download_button(
    label="Download portfolio risk summary as CSV",
    data=portfolio_results.to_csv(index=False),
    file_name="portfolio_risk_summary.csv",
    mime="text/csv"
)