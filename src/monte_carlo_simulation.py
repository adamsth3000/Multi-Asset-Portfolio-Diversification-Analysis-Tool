import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RETURNS_PATH = "data/processed/etf_returns.csv"
PORTFOLIO_PATH = "results/hrp_portfolio.csv"

N_SIMULATIONS = 5000
N_DAYS = 252 * 10


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    portfolio = pd.read_csv(PORTFOLIO_PATH)

    return returns, portfolio


def portfolio_returns(returns, portfolio):

    weights = portfolio.set_index("ETF")["Weight"]

    aligned_returns = returns[weights.index]

    port_returns = aligned_returns.dot(weights)

    return port_returns


def bootstrap_simulation(port_returns):

    simulations = []

    for i in range(N_SIMULATIONS):

        sampled = np.random.choice(
            port_returns,
            size=N_DAYS,
            replace=True
        )

        path = np.cumprod(1 + sampled)

        simulations.append(path)

    return np.array(simulations)


def parametric_simulation(port_returns):

    mu = port_returns.mean()
    sigma = port_returns.std()

    simulations = []

    for i in range(N_SIMULATIONS):

        sampled = np.random.normal(
            mu,
            sigma,
            N_DAYS
        )

        path = np.cumprod(1 + sampled)

        simulations.append(path)

    return np.array(simulations)


def plot_simulations(simulations, title, filename):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))

    for i in range(100):
        plt.plot(simulations[i], alpha=0.1)

    plt.title(title)
    plt.xlabel("Days")
    plt.ylabel("Portfolio Growth")

    plt.tight_layout()

    plt.savefig(filename)


def main():

    returns, portfolio = load_data()

    port_returns = portfolio_returns(returns, portfolio)

    print("Running bootstrap Monte Carlo...")

    boot = bootstrap_simulation(port_returns)

    plot_simulations(
        boot,
        "Bootstrap Monte Carlo Simulation",
        "results/bootstrap_monte_carlo.png"
    )

    print("Running parametric Monte Carlo...")

    param = parametric_simulation(port_returns)

    plot_simulations(
        param,
        "Parametric Monte Carlo Simulation",
        "results/parametric_monte_carlo.png"
    )

    print("Monte Carlo simulations complete")


if __name__ == "__main__":
    main()