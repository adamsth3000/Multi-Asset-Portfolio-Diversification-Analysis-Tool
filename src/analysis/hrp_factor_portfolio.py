import os
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

os.makedirs("results", exist_ok=True)

RETURNS_PATH = "data/processed/etf_returns.csv"
FACTOR_PATH = "results/factor_exposures.csv"


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    factors = pd.read_csv(FACTOR_PATH, index_col=0)

    return returns, factors


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

    return np.dot(weights.T, np.dot(cov_slice, weights))


def recursive_bisection(cov, sorted_items):

    weights = pd.Series(1.0, index=sorted_items)

    clusters = [sorted_items]

    while len(clusters) > 0:

        cluster = clusters.pop(0)

        if len(cluster) <= 1:
            continue

        split = int(len(cluster) / 2)

        c1 = cluster[:split]
        c2 = cluster[split:]

        var1 = get_cluster_var(cov, c1)
        var2 = get_cluster_var(cov, c2)

        alpha = 1 - var1 / (var1 + var2)

        weights[c1] *= alpha
        weights[c2] *= 1 - alpha

        clusters.append(c1)
        clusters.append(c2)

    return weights


def main():

    print("Building FACTOR-AWARE HRP portfolio...")

    returns, factors = load_data()

    # align assets
    common_assets = list(set(returns.columns) & set(factors.index))
    returns = returns[common_assets]
    factors = factors.loc[common_assets]

    # base covariance
    cov = returns.cov()
    corr = returns.corr()

    # 🔥 factor penalty (core idea)
    factor_matrix = factors[["beta_mkt","beta_smb","beta_hml"]].values
    factor_cov = np.cov(factor_matrix.T)

    cov_adjusted = cov + 0.05 * np.mean(factor_cov)

    dist = correl_distance(corr)

    link = linkage(squareform(dist), method="single")

    sorted_ix = get_quasi_diag(link)
    sorted_tickers = returns.columns[sorted_ix]

    weights = recursive_bisection(cov_adjusted, sorted_tickers)
    weights = weights / weights.sum()

    portfolio = pd.DataFrame({
        "ETF": weights.index,
        "Weight": weights.values
    }).sort_values("Weight", ascending=False)

    portfolio.to_csv("results/hrp_portfolio_factored.csv", index=False)

    print("\nFactor HRP portfolio created\n")
    print(portfolio.head(15))


if __name__ == "__main__":
    main()
