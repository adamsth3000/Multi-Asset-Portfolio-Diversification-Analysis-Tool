import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

RETURNS_PATH = "data/processed/etf_returns.csv"
REGIME_PATH = "results/regime_portfolio_analysis.csv"

DATASETS = {
    "Traditional": "results/portfolio_simulations.csv",
    "Factor": "results/factor_portfolio_simulations.csv",
    "Combination": "results/combination_portfolio_simulations.csv"
}


@st.cache_data
def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    regime_df = pd.read_csv(REGIME_PATH)

    datasets = {}
    for key, path in DATASETS.items():
        try:
            datasets[key] = pd.read_csv(path)
        except:
            datasets[key] = pd.DataFrame()

    return returns, regime_df, datasets


def get_portfolio_details(row, datasets):

    dataset = datasets[row["Type"]]

    match = dataset[dataset["PortfolioID"] == row["PortfolioID"]]

    if match.empty:
        return None, None

    match = match.iloc[0]

    if "ETFs" in match:
        assets = match["ETFs"].split(",")
    elif "Assets" in match:
        assets = match["Assets"].split(",")
    else:
        return None, None

    weights = np.array([float(x) for x in match["Weights"].split(",")])

    return assets, weights


def compute_stats(returns, assets, weights):

    subset = returns[assets]

    port_returns = subset.dot(weights)

    annual_return = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)

    sharpe = annual_return / volatility if volatility > 0 else 0

    return annual_return, volatility, sharpe


def main():

    st.title("Top Portfolios — Annual Return vs Avg Sharpe")

    returns, regime_df, datasets = load_data()

    st.sidebar.header("Filters")

    top_n = st.sidebar.slider("Top N Portfolios", 10, 500, 100)

    selected_types = st.sidebar.multiselect(
        "Portfolio Types",
        regime_df["Type"].unique(),
        default=regime_df["Type"].unique()
    )

    df = regime_df[regime_df["Type"].isin(selected_types)]

    # Use robustness only to select top candidates
    df = df.sort_values("Robustness", ascending=False).head(top_n)

    # Compute stats
    stats = []

    for _, row in df.iterrows():

        assets, weights = get_portfolio_details(row, datasets)

        if assets is None:
            continue

        r, v, s = compute_stats(returns, assets, weights)

        stats.append({
            "PortfolioID": row["PortfolioID"],
            "Type": row["Type"],
            "AnnualReturn": r,
            "Volatility": v,
            "Sharpe": s,
            "Avg_Sharpe": row["Avg_Sharpe"]
        })

    stats_df = pd.DataFrame(stats)

    st.subheader("Annual Return vs Average Sharpe")

    fig = px.scatter(
        stats_df,
        x="Avg_Sharpe",
        y="AnnualReturn",
        color="Type",
        hover_data=["PortfolioID"],
        title="Annual Return vs Average Sharpe (Across Regimes)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Select Portfolio")

    selected_id = st.selectbox(
        "Choose Portfolio ID",
        stats_df["PortfolioID"]
    )

    selected_row = stats_df[stats_df["PortfolioID"] == selected_id].iloc[0]

    original = df[df["PortfolioID"] == selected_id].iloc[0]

    assets, weights = get_portfolio_details(original, datasets)

    alloc_df = pd.DataFrame({
        "Asset": assets,
        "Weight": weights
    }).sort_values("Weight", ascending=False)

    st.subheader("Portfolio Allocation")

    st.dataframe(alloc_df)

    fig2 = px.bar(
        alloc_df,
        x="Asset",
        y="Weight",
        title="Asset Allocation"
    )

    st.plotly_chart(fig2, use_container_width=True)


if __name__ == "__main__":
    main()

