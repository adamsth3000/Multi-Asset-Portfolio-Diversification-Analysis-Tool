import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
FACTOR_PATH = "results/factor_exposures.csv"

PORTFOLIOS = {
    "HRP_Factored": "results/hrp_portfolio_factored.csv"
}


def load_returns():
    return pd.read_csv(RETURNS_PATH, index_col=0, parse_dates=True)


def load_factors():
    return pd.read_csv(FACTOR_PATH, index_col=0)


def portfolio_returns(returns, portfolio):

    weights = portfolio.set_index("ETF")["Weight"]
    subset = returns[weights.index]

    return subset.dot(weights)


def compute_metrics(port_returns):

    annual_return = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility

    return annual_return, volatility, sharpe


def factor_exposure(portfolio, factors):

    weights = portfolio.set_index("ETF")["Weight"]

    subset = factors.loc[weights.index]

    beta = np.dot(weights, subset[["beta_mkt","beta_smb","beta_hml"]].values)

    r2 = np.average(subset["r_squared"], weights=weights)

    return beta, r2


def main():

    returns = load_returns()
    factors = load_factors()

    results = []

    for name, path in PORTFOLIOS.items():

        portfolio = pd.read_csv(path)

        port_returns = portfolio_returns(returns, portfolio)

        ret, vol, sharpe = compute_metrics(port_returns)

        beta, r2 = factor_exposure(portfolio, factors)

        results.append({
            "Portfolio": name,
            "Return": ret,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Beta_MKT": beta[0],
            "Beta_SMB": beta[1],
            "Beta_HML": beta[2],
            "Avg_R2": r2,
            "FactorMagnitude": np.linalg.norm(beta)
        })

    df = pd.DataFrame(results)

    df.to_csv("results/portfolio_metrics_factored.csv", index=False)

    print(df)


if __name__ == "__main__":
    main()
