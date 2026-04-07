import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

RETURNS_PATH = "data/processed/etf_returns.csv"
SIM_PATH = "results/factor_portfolio_simulations.csv"


@st.cache_data
def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    sims = pd.read_csv(SIM_PATH)

    return returns, sims


def portfolio_returns(returns, etfs, weights):

    subset = returns[etfs]

    port_returns = subset.dot(weights)

    return port_returns


def portfolio_metrics(port_returns):

    annual_return = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)

    sharpe = annual_return / volatility

    cumulative = (1 + port_returns).cumprod()

    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()

    max_dd = drawdown.max()

    return annual_return, volatility, sharpe, max_dd


def main():

    st.title("Efficient Frontier — Factor-Aware Portfolio Explorer")

    returns, sims = load_data()

    st.sidebar.header("Filters")

    min_sharpe = st.sidebar.slider(
        "Minimum Sharpe",
        float(sims["Sharpe"].min()),
        float(sims["Sharpe"].max()),
        float(sims["Sharpe"].min())
    )

    max_vol = st.sidebar.slider(
        "Maximum Volatility",
        float(sims["Volatility"].min()),
        float(sims["Volatility"].max()),
        float(sims["Volatility"].max())
    )

    filtered = sims[
        (sims["Sharpe"] >= min_sharpe) &
        (sims["Volatility"] <= max_vol)
    ]

    st.subheader("Efficient Frontier")

    fig = px.scatter(
        filtered,
        x="Volatility",
        y="AnnualReturn",
        color="Sharpe",
        hover_data={
            "PortfolioID": True,
            "AnnualReturn": True,
            "Volatility": True,
            "Sharpe": True,
            "ETFs": True
        },
        title="Factor-Aware Efficient Frontier"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Select Portfolio(s)")

    selected_ids = st.multiselect(
        "Choose portfolio IDs",
        filtered["PortfolioID"].tolist()
    )

    if selected_ids:

        for pid in selected_ids:

            st.divider()

            st.header(f"Portfolio {pid}")

            row = sims[sims["PortfolioID"] == pid].iloc[0]

            etfs = row["ETFs"].split(",")

            weights = np.array(
                [float(x) for x in row["Weights"].split(",")]
            )

            allocation = pd.DataFrame({
                "ETF": etfs,
                "Weight": weights
            })

            st.subheader("ETF Allocation")

            st.dataframe(allocation)

            port_returns = portfolio_returns(
                returns,
                etfs,
                weights
            )

            r, v, s, dd = portfolio_metrics(port_returns)

            metrics = pd.DataFrame({
                "Annual Return": [r],
                "Volatility": [v],
                "Sharpe Ratio": [s],
                "Max Drawdown": [dd]
            })

            st.subheader("Portfolio Metrics")

            st.dataframe(metrics)

            growth = (1 + port_returns).cumprod()

            st.subheader("Growth of $1")

            st.line_chart(growth)


if __name__ == "__main__":
    main()
