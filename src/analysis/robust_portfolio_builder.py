import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
COMBO_PATH = "results/portfolio_combination_summary.csv"

TOP_K = 10  # number of top combinations


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    combos = pd.read_csv(COMBO_PATH)

    return returns, combos


def optimize_equal_weight(assets):

    n = len(assets)
    return np.ones(n) / n


def build_portfolio(returns, assets):

    weights = optimize_equal_weight(assets)

    subset = returns[assets]

    port_returns = subset.dot(weights)

    annual_return = port_returns.mean() * 252
    volatility = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0

    return {
        "Assets": ",".join(assets),
        "Return": annual_return,
        "Volatility": volatility,
        "Sharpe": sharpe
    }


def main():

    print("Building robust portfolios from top combinations...")

    returns, combos = load_data()

    # Ensure TotalFrequency exists
    if "TotalFrequency" not in combos.columns:
        combos["TotalFrequency"] = (
            combos["Num_Traditional_Portfolios"] +
            combos["Num_Factor_Portfolios"]
        )

    # Select top combinations
    top = combos.sort_values(
        ["Avg_Sharpe", "TotalFrequency"],
        ascending=False
    ).head(TOP_K)

    results = []

    for _, row in top.iterrows():

        assets = row["OverlapAssets"].split(",")

        n = len(assets)
        weights = np.ones(n) / n

        subset = returns[assets]
        port_returns = subset.dot(weights)

        annual_return = port_returns.mean() * 252
        volatility = port_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0

        results.append({
            "Assets": ",".join(assets),
            "Return": annual_return,
            "Volatility": volatility,
            "Sharpe": sharpe
        })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/robust_portfolios.csv",
        index=False
    )

    print("\nRobust portfolios built:\n")
    print(df)



if __name__ == "__main__":
    main()
