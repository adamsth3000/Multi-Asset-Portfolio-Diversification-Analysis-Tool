import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
CLUSTER_PATH = "results/etf_clusters.csv"
FACTOR_PATH = "results/factor_exposures.csv"
SIMILARITY_PATH = "results/factor_similarity_matrix.csv"


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    clusters = pd.read_csv(CLUSTER_PATH)

    factors = pd.read_csv(FACTOR_PATH, index_col=0)

    similarity = pd.read_csv(SIMILARITY_PATH, index_col=0)

    return returns, clusters, factors, similarity


def compute_volatility(returns):
    return returns.std()


def compute_factor_uniqueness(asset, cluster_assets, similarity):

    distances = []

    for other in cluster_assets:
        if other == asset:
            continue

        if asset in similarity.index and other in similarity.columns:
            distances.append(similarity.loc[asset, other])

    if len(distances) == 0:
        return 0

    return np.mean(distances)


def score_asset(asset, vol, factors, similarity, cluster_assets):

    if asset not in factors.index:
        return -np.inf

    volatility = vol[asset]
    r2 = factors.loc[asset, "r_squared"]

    uniqueness = compute_factor_uniqueness(
        asset,
        cluster_assets,
        similarity
    )

    # Tunable scoring system
    score = (
        -1.0 * volatility
        -0.5 * r2
        +1.5 * uniqueness
    )

    return score


def select_representatives(returns, clusters, factors, similarity):

    vol = compute_volatility(returns)

    selected = []

    for cluster_id in clusters["Cluster"].unique():

        assets = clusters[
            clusters["Cluster"] == cluster_id
        ]["ETF"].tolist()

        best_asset = None
        best_score = -np.inf

        for asset in assets:

            score = score_asset(
                asset,
                vol,
                factors,
                similarity,
                assets
            )

            if score > best_score:
                best_score = score
                best_asset = asset

        if best_asset:
            selected.append(best_asset)

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

    print("Building FACTOR-AWARE cluster portfolio...")

    returns, clusters, factors, similarity = load_data()

    selected_assets = select_representatives(
        returns,
        clusters,
        factors,
        similarity
    )

    portfolio = build_equal_weight_portfolio(selected_assets)

    portfolio.to_csv(
        "results/factor_cluster_portfolio.csv",
        index=False
    )

    print("\nFactor-aware cluster portfolio created\n")
    print(portfolio)


if __name__ == "__main__":
    main()
