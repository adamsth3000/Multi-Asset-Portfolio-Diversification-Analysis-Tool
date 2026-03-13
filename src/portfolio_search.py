import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
N_PORTFOLIOS = 10000


def load_returns():

    print("Loading ETF return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def compute_covariance(returns):

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


def random_weights(n):

    weights = np.random.random(n)

    weights /= np.sum(weights)

    return weights


def search_portfolios(vol, cov, tickers):

    results = []

    n_assets = len(tickers)

    for i in range(N_PORTFOLIOS):

        weights = random_weights(n_assets)

        dr = diversification_ratio(weights, vol.values, cov.values)

        results.append((dr, weights))

        if i % 1000 == 0:
            print(f"Evaluated {i} portfolios")

    return results


def extract_best(results, tickers):

    results.sort(reverse=True)

    best_dr, best_weights = results[0]

    portfolio = pd.DataFrame({
        "ETF": tickers,
        "Weight": best_weights
    })

    portfolio = portfolio.sort_values(
        "Weight",
        ascending=False
    )

    portfolio.to_csv(
        "results/best_diversified_portfolio.csv",
        index=False
    )

    print("\nBest diversification ratio:", best_dr)

    print("\nTop allocations:")
    print(portfolio.head(10))


def main():

    returns = load_returns()

    cov = compute_covariance(returns)

    vol = compute_volatility(returns)

    results = search_portfolios(
        vol,
        cov,
        returns.columns
    )

    extract_best(results, returns.columns)


if __name__ == "__main__":
    main()