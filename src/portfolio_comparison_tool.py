import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RETURNS_PATH = "data/processed/etf_returns.csv"
SIM_PATH = "results/portfolio_simulations.csv"

HRP_PATH = "results/hrp_portfolio.csv"
CLUSTER_PATH = "results/cluster_diversified_portfolio.csv"


def load_returns():
    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )
    return returns


def load_simulation_portfolio(pid):
    sims = pd.read_csv(SIM_PATH)

    row = sims[sims["PortfolioID"] == pid].iloc[0]

    etfs = row["ETFs"].split(",")
    weights = np.array([float(x) for x in row["Weights"].split(",")])

    return etfs, weights


def load_algorithmic_portfolio(path):
    df = pd.read_csv(path)

    etfs = df["ETF"].tolist()
    weights = df["Weight"].values

    return etfs, weights


def portfolio_returns(returns, etfs, weights):
    subset = returns[etfs]
    port_returns = subset.dot(weights)
    return port_returns


def plot_growth(returns, portfolios):
    plt.figure(figsize=(10, 6))

    for name, (etfs, weights) in portfolios.items():
        port_returns = portfolio_returns(returns, etfs, weights)
        growth = (1 + port_returns).cumprod()
        plt.plot(growth, label=name)

    plt.title("Portfolio Growth Comparison")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()

    plt.tight_layout()

    plt.savefig("results/portfolio_growth_comparison_custom.png")

    print("\nSaved chart to results/portfolio_growth_comparison_custom.png")

    plt.show()


def get_portfolio_ids():

    print("\nEnter Portfolio IDs to analyze.")
    print("Press ENTER without typing anything to finish.\n")

    ids = []

    while True:
        user_input = input("Portfolio ID: ")

        if user_input.strip() == "":
            break

        try:
            ids.append(int(user_input))
        except ValueError:
            print("Invalid ID. Please enter a number.")

    return ids


def main():

    portfolio_ids = get_portfolio_ids()

    if len(portfolio_ids) == 0:
        print("No portfolios selected. Exiting.")
        return

    returns = load_returns()

    portfolios = {}

    for pid in portfolio_ids:
        etfs, weights = load_simulation_portfolio(pid)
        portfolios[f"Portfolio ID {pid}"] = (etfs, weights)

    hrp = load_algorithmic_portfolio(HRP_PATH)
    cluster = load_algorithmic_portfolio(CLUSTER_PATH)

    portfolios["HRP"] = hrp
    portfolios["Cluster"] = cluster

    plot_growth(returns, portfolios)


if __name__ == "__main__":
    main()