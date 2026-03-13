import pandas as pd
import numpy as np

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


def compute_metrics(port_returns):

    annual_return = port_returns.mean() * 252

    volatility = port_returns.std() * np.sqrt(252)

    sharpe = annual_return / volatility

    cumulative = (1 + port_returns).cumprod()

    drawdown = (cumulative.cummax() - cumulative) / cumulative.cummax()

    max_drawdown = drawdown.max()

    return annual_return, volatility, sharpe, max_drawdown


def main():

    returns = load_returns()

    results = []

    for name, path in PORTFOLIOS.items():

        portfolio = pd.read_csv(path)

        port_returns = portfolio_returns(returns, portfolio)

        annual_return, volatility, sharpe, max_dd = compute_metrics(port_returns)

        results.append({
            "Portfolio": name,
            "Annual Return": annual_return,
            "Annual Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/portfolio_comparison_metrics.csv",
        index=False
    )

    print("\nPortfolio Comparison Metrics\n")
    print(df)

    print("\nSaved to results/portfolio_comparison_metrics.csv")


if __name__ == "__main__":
    main()