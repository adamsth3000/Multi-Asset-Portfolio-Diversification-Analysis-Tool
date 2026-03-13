import pandas as pd
import numpy as np
import random

RETURNS_PATH = "data/processed/etf_returns.csv"

N_PORTFOLIOS = 5000
MIN_ASSETS = 4
MAX_ASSETS = 12


def load_returns():
    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )
    return returns


def simulate_portfolios(returns):

    tickers = list(returns.columns)

    results = []

    for i in range(N_PORTFOLIOS):

        n_assets = random.randint(MIN_ASSETS, MAX_ASSETS)

        selected = random.sample(tickers, n_assets)

        weights = np.random.random(n_assets)

        weights /= weights.sum()

        subset_returns = returns[selected]

        port_returns = subset_returns.dot(weights)

        annual_return = port_returns.mean() * 252

        volatility = port_returns.std() * np.sqrt(252)

        sharpe = annual_return / volatility if volatility > 0 else 0

        results.append({
    "PortfolioID": i,
    "AnnualReturn": annual_return,
    "Volatility": volatility,
    "Sharpe": sharpe,
    "ETFs": ",".join(selected),
    "Weights": ",".join([str(w) for w in weights])
})

    df = pd.DataFrame(results)

    df.to_csv(
        "results/portfolio_simulations.csv",
        index=False
    )

    print("Portfolio simulations saved")


def main():

    returns = load_returns()

    simulate_portfolios(returns)


if __name__ == "__main__":
    main()