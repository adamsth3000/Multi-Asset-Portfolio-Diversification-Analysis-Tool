import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
CLUSTER_PATH = "results/etf_clusters.csv"


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    clusters = pd.read_csv(CLUSTER_PATH)

    return returns, clusters


def compute_volatility(returns):

    return returns.std()


def select_representatives(returns, clusters):

    vol = compute_volatility(returns)

    selected = []

    for cluster_id in clusters["Cluster"].unique():

        assets = clusters[clusters["Cluster"] == cluster_id]["ETF"]

        asset_vol = vol[assets]

        representative = asset_vol.idxmin()

        selected.append(representative)

    return selected


def build_equal_weight_portfolio(assets):

    n = len(assets)

    weights = np.ones(n) / n

    portfolio = pd.DataFrame({
        "ETF": assets,
        "Weight": weights
    })

    return portfolio


def main():

    returns, clusters = load_data()

    selected_assets = select_representatives(
        returns,
        clusters
    )

    portfolio = build_equal_weight_portfolio(
        selected_assets
    )

    portfolio.to_csv(
        "results/cluster_diversified_portfolio.csv",
        index=False
    )

    print("\nCluster diversified portfolio created\n")

    print(portfolio)


if __name__ == "__main__":
    main()