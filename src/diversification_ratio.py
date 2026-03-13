import pandas as pd
import numpy as np


RETURNS_PATH = "data/processed/etf_returns.csv"


def load_returns():

    print("Loading ETF return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def compute_covariance(returns):

    print("Computing covariance matrix...")

    cov = returns.cov()

    return cov


def compute_volatility(returns):

    vol = returns.std()

    return vol


def diversification_ratio(weights, vol, cov):

    weighted_vol = np.dot(weights, vol)

    portfolio_vol = np.sqrt(
        np.dot(weights.T, np.dot(cov, weights))
    )

    return weighted_vol / portfolio_vol


def equal_weight_portfolio(n):

    weights = np.ones(n) / n

    return weights


def main():

    returns = load_returns()

    cov = compute_covariance(returns)

    vol = compute_volatility(returns)

    n_assets = len(vol)

    weights = equal_weight_portfolio(n_assets)

    dr = diversification_ratio(weights, vol.values, cov.values)

    print("\nDiversification Ratio:", dr)


if __name__ == "__main__":
    main()