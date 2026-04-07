import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

RETURNS_PATH = "data/processed/etf_returns.csv"
COMBO_PATH = "results/portfolio_combination_summary.csv"


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    combos = pd.read_csv(COMBO_PATH)

    return returns, combos


def correl_distance(corr):
    return np.sqrt((1 - corr) / 2)


def get_quasi_diag(link):

    link = link.astype(int)

    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:

        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)

        df0 = sort_ix[sort_ix >= num_items]

        i = df0.index
        j = df0.values - num_items

        sort_ix[i] = link[j, 0]

        df1 = pd.Series(link[j, 1], index=i + 1)

        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def get_cluster_var(cov, cluster_items):

    cov_slice = cov.loc[cluster_items, cluster_items]

    weights = np.ones(len(cluster_items)) / len(cluster_items)

    variance = np.dot(weights.T, np.dot(cov_slice, weights))

    return variance


def recursive_bisection(cov, sorted_items):

    weights = pd.Series(1.0, index=sorted_items)

    clusters = [sorted_items]

    while len(clusters) > 0:

        cluster = clusters.pop(0)

        if len(cluster) <= 1:
            continue

        split = int(len(cluster) / 2)

        cluster1 = cluster[:split]
        cluster2 = cluster[split:]

        var1 = get_cluster_var(cov, cluster1)
        var2 = get_cluster_var(cov, cluster2)

        alpha = 1 - var1 / (var1 + var2)

        weights[cluster1] *= alpha
        weights[cluster2] *= 1 - alpha

        clusters.append(cluster1)
        clusters.append(cluster2)

    return weights


def main():

    print("Building HRP combination portfolio...")

    returns, combos = load_data()

    # Ensure TotalFrequency exists
    if "TotalFrequency" not in combos.columns:
        combos["TotalFrequency"] = (
            combos["Num_Traditional_Portfolios"] +
            combos["Num_Factor_Portfolios"]
        )

    # Use best combination
    best = combos.sort_values(
        ["Avg_Sharpe", "TotalFrequency"],
        ascending=False
    ).iloc[0]

    assets = best["OverlapAssets"].split(",")

    subset = returns[assets]

    cov = subset.cov()
    corr = subset.corr()

    dist = correl_distance(corr)

    link = linkage(squareform(dist), method="single")

    sorted_ix = get_quasi_diag(link)
    sorted_assets = subset.columns[sorted_ix]

    weights = recursive_bisection(cov, sorted_assets)
    weights = weights / weights.sum()

    portfolio = pd.DataFrame({
        "ETF": weights.index,
        "Weight": weights.values
    })

    portfolio = portfolio.sort_values(
        "Weight",
        ascending=False
    )

    portfolio.to_csv(
        "results/hrp_combination_portfolio.csv",
        index=False
    )

    print("\nHRP Combination Portfolio:\n")
    print(portfolio)


if __name__ == "__main__":
    main()
