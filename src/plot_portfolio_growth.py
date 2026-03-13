import pandas as pd
import matplotlib.pyplot as plt

RETURNS_PATH = "data/processed/etf_returns.csv"

PORTFOLIOS = {
    "HRP": "results/hrp_portfolio.csv",
    "Cluster": "results/cluster_diversified_portfolio.csv",
    "RandomSearch": "results/best_diversified_portfolio.csv"
}


def load_returns():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def portfolio_returns(returns, portfolio):

    weights = portfolio.set_index("ETF")["Weight"]

    aligned_returns = returns[weights.index]

    port_returns = aligned_returns.dot(weights)

    return port_returns


def main():

    returns = load_returns()

    plt.figure(figsize=(12,6))

    for name, path in PORTFOLIOS.items():

        portfolio = pd.read_csv(path)

        port_returns = portfolio_returns(returns, portfolio)

        growth = (1 + port_returns).cumprod()

        plt.plot(growth, label=name)

    plt.title("Portfolio Growth Comparison")

    plt.xlabel("Date")

    plt.ylabel("Growth of $1")

    plt.legend()

    plt.tight_layout()

    plt.savefig("results/portfolio_growth_comparison.png")

    print("Portfolio growth chart saved")


if __name__ == "__main__":
    main()