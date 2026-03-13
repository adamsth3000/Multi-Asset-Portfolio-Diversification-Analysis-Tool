import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"

WINDOW = 756   # roughly 3 years of trading days


def load_returns():

    print("Loading return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def compute_covariance(data):

    return data.cov()


def compute_volatility(data):

    return data.std()


def diversification_ratio(weights, vol, cov):

    weighted_vol = np.dot(weights, vol)

    portfolio_vol = np.sqrt(
        np.dot(weights.T, np.dot(cov, weights))
    )

    return weighted_vol / portfolio_vol


def equal_weights(n):

    return np.ones(n) / n


def rolling_analysis(returns):

    results = []

    for i in range(WINDOW, len(returns)):

        window_data = returns.iloc[i-WINDOW:i]

        cov = compute_covariance(window_data)

        vol = compute_volatility(window_data)

        weights = equal_weights(len(vol))

        dr = diversification_ratio(
            weights,
            vol.values,
            cov.values
        )

        date = returns.index[i]

        results.append((date, dr))

    return pd.DataFrame(results, columns=["Date","DiversificationRatio"])


def main():

    returns = load_returns()

    df = rolling_analysis(returns)

    df.to_csv(
        "results/rolling_diversification.csv",
        index=False
    )

    print("Rolling diversification analysis complete")


if __name__ == "__main__":
    main()