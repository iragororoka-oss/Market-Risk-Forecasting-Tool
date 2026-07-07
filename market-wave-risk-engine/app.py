import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from src.stress_testing import run_stress_tests
from src.simulation_engine import MarketWaveRiskEngine
from src.pytorch_vol_model import train_volatility_model
from src.stress_testing import simulate_paths_with_params
from src.backtesting import run_var_backtest
from src.historical_simulation import historical_var_es, historical_boundary_probability
from src.regime_detection import regime_adjusted_risk

st.set_page_config(
    page_title="Market Wave Risk Forecasting Tool",
    layout="wide"
)

st.title("Market Wave Risk Forecasting Tool")

st.write(
    """
    This tool uses historical market data, Monte Carlo simulation, and risk metrics
    to estimate possible future price paths, boundary probabilities, VaR, and Expected Shortfall.
    """
)


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Model Inputs")

ticker = st.sidebar.text_input("Ticker", value="AAPL")

period = st.sidebar.selectbox(
    "Historical period",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

horizon_days = st.sidebar.slider(
    "Simulation horizon (days)",
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


# --------------------------------------------------
# Load market data
# --------------------------------------------------

data = yf.download(ticker, period=period, auto_adjust=True)

if data.empty:
    st.error("No data found. Try another ticker.")
    st.stop()

prices = data["Close"].dropna()
if isinstance(prices, pd.DataFrame):
    prices = prices.iloc[:, 0]
prices = prices.dropna().astype(float)
engine = MarketWaveRiskEngine(prices)

paths = engine.simulate_paths(
    horizon_days=horizon_days,
    n_sims=n_sims,
    seed=7
)

current_price = float(prices.iloc[-1])
upper_boundary = current_price * (1 + profit_target)
lower_boundary = current_price * (1 - loss_limit)

boundary_results = engine.boundary_probabilities(
    paths,
    upper_boundary,
    lower_boundary
)

risk_95 = engine.tail_risk(paths, confidence=0.95)
risk_99 = engine.tail_risk(paths, confidence=0.99)


# --------------------------------------------------
# Display metrics
# --------------------------------------------------

st.subheader(f"Results for {ticker.upper()}")

col1, col2, col3 = st.columns(3)

col1.metric("Latest Price", f"${current_price:.2f}")
col2.metric("Daily Drift", f"{engine.mu_daily:.4%}")
col3.metric("Daily Volatility", f"{engine.sigma_daily:.4%}")

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
# Plot simulated paths
# --------------------------------------------------

st.subheader("Simulated Market Waves")

fig, ax = plt.subplots(figsize=(10, 6))

for i in range(min(100, paths.shape[1])):
    ax.plot(paths[:, i], linewidth=0.8, alpha=0.5)

ax.axhline(upper_boundary, linestyle="--", label="Profit Boundary")
ax.axhline(lower_boundary, linestyle="--", label="Loss Boundary")

ax.set_title("Monte Carlo Simulated Price Paths")
ax.set_xlabel("Days")
ax.set_ylabel("Simulated Price")
ax.legend()

st.pyplot(fig)


# --------------------------------------------------
# Historical price chart
# --------------------------------------------------

st.subheader("Historical Price Data")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(prices)
ax2.set_title(f"{ticker.upper()} Historical Price")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")

st.pyplot(fig2)

# --------------------------------------------------
# Stress Testing
# --------------------------------------------------

st.subheader("Stress Test Scenario Comparison")

stress_results = run_stress_tests(
    engine=engine,
    horizon_days=horizon_days,
    n_sims=n_sims,
    upper_boundary=upper_boundary,
    lower_boundary=lower_boundary,
    seed=11
)

stress_display = stress_results.copy()

percent_columns = [
    "Daily Drift",
    "Daily Volatility",
    "Initial Shock",
    "Profit First",
    "Loss First",
    "Neither Boundary",
    "95% VaR",
    "95% Expected Shortfall",
    "99% VaR",
    "99% Expected Shortfall"
]

for col in percent_columns:
    stress_display[col] = stress_display[col].map(lambda x: f"{x:.2%}")

st.dataframe(
    stress_display,
    use_container_width=True
)

csv_data = stress_results.to_csv(index=False)

st.download_button(
    label="Download stress test results as CSV",
    data=csv_data,
    file_name=f"{ticker.upper()}_stress_test_results.csv",
    mime="text/csv"
)

# --------------------------------------------------
# PyTorch Volatility Forecast Model
# --------------------------------------------------

st.subheader("Model Comparison: Historical Volatility vs PyTorch Volatility Forecast")

try:
    ml_result = train_volatility_model(
        returns=engine.returns,
        lookback=30,
        forecast_window=10,
        epochs=300,
        learning_rate=0.001
    )

    predicted_sigma = ml_result["predicted_volatility"]

    ml_paths = simulate_paths_with_params(
        s0=current_price,
        mu_daily=engine.mu_daily,
        sigma_daily=predicted_sigma,
        horizon_days=horizon_days,
        n_sims=n_sims,
        seed=21
    )

    ml_boundary_results = engine.boundary_probabilities(
        ml_paths,
        upper_boundary,
        lower_boundary
    )

    ml_risk_95 = engine.tail_risk(ml_paths, confidence=0.95)
    ml_risk_99 = engine.tail_risk(ml_paths, confidence=0.99)

    comparison_table = pd.DataFrame([
        {
            "Model": "Historical Volatility Monte Carlo",
            "Daily Volatility": engine.sigma_daily,
            "Profit First": boundary_results["prob_hit_profit_first"],
            "Loss First": boundary_results["prob_hit_loss_first"],
            "95% VaR": risk_95["VaR"],
            "95% Expected Shortfall": risk_95["Expected Shortfall"],
            "99% VaR": risk_99["VaR"],
            "99% Expected Shortfall": risk_99["Expected Shortfall"]
        },
        {
            "Model": "PyTorch Volatility Forecast Monte Carlo",
            "Daily Volatility": predicted_sigma,
            "Profit First": ml_boundary_results["prob_hit_profit_first"],
            "Loss First": ml_boundary_results["prob_hit_loss_first"],
            "95% VaR": ml_risk_95["VaR"],
            "95% Expected Shortfall": ml_risk_95["Expected Shortfall"],
            "99% VaR": ml_risk_99["VaR"],
            "99% Expected Shortfall": ml_risk_99["Expected Shortfall"]
        }
    ])

    comparison_display = comparison_table.copy()

    percent_cols = [
        "Daily Volatility",
        "Profit First",
        "Loss First",
        "95% VaR",
        "95% Expected Shortfall",
        "99% VaR",
        "99% Expected Shortfall"
    ]

    for col in percent_cols:
        comparison_display[col] = comparison_display[col].map(lambda x: f"{x:.2%}")

    st.dataframe(comparison_display, use_container_width=True)

    st.write(
        f"PyTorch predicted daily volatility: **{predicted_sigma:.2%}**"
    )

    st.write(
        f"Volatility model test RMSE: **{ml_result['test_rmse']:.4%}**"
    )

except ValueError as error:
    st.warning(str(error))

# --------------------------------------------------
# VaR Backtesting
# --------------------------------------------------

st.subheader("VaR Backtesting")

try:
    backtest_95_summary, backtest_95_table = run_var_backtest(
        returns=engine.returns,
        confidence=0.95,
        rolling_window=252,
        n_sims=10000,
        seed=42
    )

    backtest_99_summary, backtest_99_table = run_var_backtest(
        returns=engine.returns,
        confidence=0.99,
        rolling_window=252,
        n_sims=10000,
        seed=99
    )

    backtest_summary_df = pd.DataFrame([
        backtest_95_summary,
        backtest_99_summary
    ])

    display_backtest = backtest_summary_df.copy()

    percent_cols = [
        "Confidence Level",
        "Expected Breach Rate",
        "Observed Breach Rate",
        "Average VaR",
        "Maximum Realized Loss"
    ]

    for col in percent_cols:
        display_backtest[col] = display_backtest[col].map(lambda x: f"{x:.2%}")

    display_backtest["Expected Breaches"] = display_backtest["Expected Breaches"].map(lambda x: f"{x:.2f}")

    st.dataframe(display_backtest, use_container_width=True)

    st.write(
        """
        A VaR breach occurs when the realized loss is larger than the VaR estimate.
        For example, a 95% VaR model should breach roughly 5% of the time.
        A 99% VaR model should breach roughly 1% of the time.
        """
    )

except ValueError as error:
    st.warning(str(error))

# --------------------------------------------------
# Historical Simulation Risk Model
# --------------------------------------------------

st.subheader("Model Comparison: Monte Carlo vs Historical Simulation")

try:
    hist_95 = historical_var_es(
        returns=engine.returns,
        confidence=0.95
    )

    hist_99 = historical_var_es(
        returns=engine.returns,
        confidence=0.99
    )

    hist_boundary = historical_boundary_probability(
        returns=engine.returns,
        current_price=current_price,
        upper_boundary=upper_boundary,
        lower_boundary=lower_boundary,
        horizon_days=horizon_days,
        n_sims=n_sims,
        seed=31
    )

    model_comparison = pd.DataFrame([
        {
            "Model": "Monte Carlo",
            "Profit First": boundary_results["prob_hit_profit_first"],
            "Loss First": boundary_results["prob_hit_loss_first"],
            "95% VaR": risk_95["VaR"],
            "95% Expected Shortfall": risk_95["Expected Shortfall"],
            "99% VaR": risk_99["VaR"],
            "99% Expected Shortfall": risk_99["Expected Shortfall"]
        },
        {
            "Model": "Historical Simulation",
            "Profit First": hist_boundary["prob_hit_profit_first"],
            "Loss First": hist_boundary["prob_hit_loss_first"],
            "95% VaR": hist_95["Historical VaR"],
            "95% Expected Shortfall": hist_95["Historical Expected Shortfall"],
            "99% VaR": hist_99["Historical VaR"],
            "99% Expected Shortfall": hist_99["Historical Expected Shortfall"]
        }
    ])

    model_display = model_comparison.copy()

    percent_cols = [
        "Profit First",
        "Loss First",
        "95% VaR",
        "95% Expected Shortfall",
        "99% VaR",
        "99% Expected Shortfall"
    ]

    for col in percent_cols:
        model_display[col] = model_display[col].map(lambda x: f"{x:.2%}")

    st.dataframe(model_display, use_container_width=True)

    st.write(
        """
        Historical simulation uses actual past returns instead of assuming a normal distribution.
        This helps compare the Monte Carlo model against a more empirical risk estimate.
        """
    )

except ValueError as error:
    st.warning(str(error))

# --------------------------------------------------
# Volatility Regime Detection
# --------------------------------------------------

st.subheader("Volatility Regime Detection")

try:
    regime_result = regime_adjusted_risk(
        engine=engine,
        horizon_days=horizon_days,
        n_sims=n_sims,
        upper_boundary=upper_boundary,
        lower_boundary=lower_boundary,
        lookback=30,
        seed=123
    )

    regime_summary = regime_result["regime_summary"]
    regime_boundary = regime_result["boundary_results"]
    regime_risk_95 = regime_result["risk_95"]
    regime_risk_99 = regime_result["risk_99"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Regime",
        regime_summary["Current Regime"]
    )

    col2.metric(
        "Current Rolling Volatility",
        f"{regime_summary['Current Rolling Volatility']:.2%}"
    )

    col3.metric(
        "Adjusted Daily Volatility",
        f"{regime_result['adjusted_sigma']:.2%}"
    )

    regime_comparison = pd.DataFrame([
        {
            "Model": "Base Monte Carlo",
            "Daily Volatility": engine.sigma_daily,
            "Profit First": boundary_results["prob_hit_profit_first"],
            "Loss First": boundary_results["prob_hit_loss_first"],
            "95% VaR": risk_95["VaR"],
            "99% VaR": risk_99["VaR"]
        },
        {
            "Model": "Regime-Adjusted Monte Carlo",
            "Daily Volatility": regime_result["adjusted_sigma"],
            "Profit First": regime_boundary["prob_hit_profit_first"],
            "Loss First": regime_boundary["prob_hit_loss_first"],
            "95% VaR": regime_risk_95["VaR"],
            "99% VaR": regime_risk_99["VaR"]
        }
    ])

    regime_display = regime_comparison.copy()

    percent_cols = [
        "Daily Volatility",
        "Profit First",
        "Loss First",
        "95% VaR",
        "99% VaR"
    ]

    for col in percent_cols:
        regime_display[col] = regime_display[col].map(lambda x: f"{x:.2%}")

    st.dataframe(regime_display, use_container_width=True)

    st.write(
        """
        The regime-adjusted model changes the simulation assumptions depending on
        whether recent volatility is calm, normal, or elevated. This makes the model
        more adaptive than using one fixed historical volatility estimate.
        """
    )

except ValueError as error:
    st.warning(str(error))