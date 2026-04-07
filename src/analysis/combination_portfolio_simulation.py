import pandas as pd
import numpy as np
import random

RETURNS_PATH = "data/processed/etf_returns.csv"
COMBO_PATH = "results/portfolio_combination_summary.csv"

N_PORTFOLIOS = 5000
TOP_COMBOS = 25


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    combos = pd.read_csv(COMBO_PATH)

    return returns, combos


def random_weights(n):
    w = np.random.random(n)
    return w / w.sum()


def simulate(returns, combos):

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
    ).head(TOP_COMBOS)

    results = []

    for i in range(N_PORTFOLIOS):

        combo_row = top.sample(1).iloc[0]

        assets = combo_row["OverlapAssets"].split(",")

        weights = np.random.random(len(assets))
        weights /= weights.sum()

        subset = returns[assets]

        port_returns = subset.dot(weights)

        annual_return = port_returns.mean() * 252
        volatility = port_returns.std() * np.sqrt(252)

        sharpe = annual_return / volatility if volatility > 0 else 0

        results.append({
            "PortfolioID": i,
            "AnnualReturn": annual_return,
            "Volatility": volatility,
            "Sharpe": sharpe,
            "Assets": ",".join(assets),
            "Weights": ",".join([str(w) for w in weights])
        })

    df = pd.DataFrame(results)

    df.to_csv(
        "results/combination_portfolio_simulations.csv",
        index=False
    )

    print("Combination portfolio simulations saved")


def main():

    returns, combos = load_data()

    simulate(returns, combos)


if __name__ == "__main__":
    main()
