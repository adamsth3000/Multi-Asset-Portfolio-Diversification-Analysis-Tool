import pandas as pd
import numpy as np
import random

RETURNS_PATH = "data/processed/etf_returns.csv"
FACTOR_PATH = "results/factor_exposures.csv"

N_PORTFOLIOS = 5000
MIN_ASSETS = 4
MAX_ASSETS = 12


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    factors = pd.read_csv(FACTOR_PATH, index_col=0)

    return returns, factors


def random_weights(n):
    w = np.random.random(n)
    return w / w.sum()


def factor_penalty(weights, subset_factors):

    beta = np.dot(weights, subset_factors.values)

    magnitude = np.linalg.norm(beta)

    return magnitude


def simulate_portfolios(returns, factors):

    tickers = list(returns.columns)

    results = []

    for i in range(N_PORTFOLIOS):

        n_assets = random.randint(MIN_ASSETS, MAX_ASSETS)

        selected = random.sample(tickers, n_assets)

        weights = random_weights(n_assets)

        subset_returns = returns[selected]

        subset_factors = factors.loc[selected][
            ["beta_mkt","beta_smb","beta_hml"]
        ]

        port_returns = subset_returns.dot(weights)

        annual_return = port_returns.mean() * 252
        volatility = port_returns.std() * np.sqrt(252)

        sharpe = annual_return / volatility if volatility > 0 else 0

        factor_mag = factor_penalty(weights, subset_factors)

        score = sharpe - 0.5 * factor_mag

        results.append({
            "PortfolioID": i,
            "AnnualReturn": annual_return,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "FactorMagnitude": factor_mag,
            "Score": score,
            "ETFs": ",".join(selected),
            "Weights": ",".join([str(w) for w in weights])
        })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/factor_portfolio_simulations.csv",
        index=False
    )

    print("Factor portfolio simulations saved")


def main():

    returns, factors = load_data()

    simulate_portfolios(returns, factors)


if __name__ == "__main__":
    main()
