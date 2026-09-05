import streamlit as st


st.set_page_config(
    page_title="Market Wave Risk Forecasting Platform",
    layout="wide"
)

st.title("Market Wave Risk Forecasting Platform")

st.write(
    """
    A financial risk forecasting platform that combines historical market data,
    Monte Carlo simulation, volatility forecasting, stress testing, backtesting,
    and portfolio-level risk analysis.
    """
)

st.subheader("Project Modules")

st.markdown(
    """
    **Single Asset Risk Dashboard**  
    Analyze one stock or ETF using Monte Carlo simulation, Value-at-Risk,
    Expected Shortfall, stress testing, historical simulation, and volatility regimes.

    **Portfolio Risk Dashboard**  
    Analyze multiple assets using portfolio weights, correlations, correlated
    Monte Carlo simulation, and asset-level risk contribution.
    """
)

st.subheader("Core Risk Metrics")

st.markdown(
    """
    - **Boundary Probability:** probability of reaching a profit target before a loss limit.
    - **Value-at-Risk:** estimated loss threshold at a selected confidence level.
    - **Expected Shortfall:** average loss beyond the VaR threshold.
    - **Stress Testing:** risk under adverse market scenarios.
    - **Backtesting:** comparison between predicted VaR breaches and realized losses.
    - **Volatility Regime Detection:** identifies calm, normal, or high-volatility market environments.
    """
)

st.info(
    "Use the sidebar to open the Single Asset Risk Dashboard or Portfolio Risk Dashboard."
)