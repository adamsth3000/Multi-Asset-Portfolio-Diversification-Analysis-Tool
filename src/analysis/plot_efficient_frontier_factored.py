import pandas as pd
import matplotlib.pyplot as plt

SIM_PATH = "results/factor_portfolio_simulations.csv"
RETURNS_PATH = "data/processed/etf_returns.csv"
HRP_PATH = "results/hrp_portfolio_factored.csv"


def load_returns():
    return pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)


def portfolio_stats(returns, portfolio):

    weights = portfolio.set_index("ETF")["Weight"]
    subset = returns[weights.index]

    port_returns = subset.dot(weights)

    annual_return = port_returns.mean() * 252
    volatility = port_returns.std() * (252 ** 0.5)

    return volatility, annual_return


def main():

    sims = pd.read_csv(SIM_PATH)
    returns = load_returns()

    plt.figure(figsize=(10,6))

    plt.scatter(
        sims["Volatility"],
        sims["AnnualReturn"],
        c=sims["Score"],
        cmap="viridis",
        alpha=0.5
    )

    # HRP overlay
    portfolio = pd.read_csv(HRP_PATH)
    vol, ret = portfolio_stats(returns, portfolio)

    plt.scatter(vol, ret, marker="*", s=200, label="HRP Factored")

    plt.xlabel("Volatility")
    plt.ylabel("Annual Return")
    plt.title("Factor-Aware Efficient Frontier")

    plt.legend()
    plt.tight_layout()

    plt.savefig("results/efficient_frontier_factored.png")

    print("Saved factor frontier")


if __name__ == "__main__":
    main()
