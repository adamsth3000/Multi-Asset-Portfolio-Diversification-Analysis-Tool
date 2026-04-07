import pandas as pd
import numpy as np

RETURNS_PATH = "data/processed/etf_returns.csv"
REGIME_PATH = "results/regime_labels.csv"

DATASETS = {
    "Traditional": "results/portfolio_simulations.csv",
    "Factor": "results/factor_portfolio_simulations.csv",
    "Combination": "results/combination_portfolio_simulations.csv"
}


def load_data():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    regimes = pd.read_csv(
        REGIME_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns, regimes


def compute_portfolio_returns(returns, etfs, weights):

    subset = returns[etfs]
    return subset.dot(weights)


def compute_metrics(series):

    if len(series) < 10:
        return 0, 0

    ann_return = series.mean() * 252
    vol = series.std() * np.sqrt(252)

    sharpe = ann_return / vol if vol > 0 else 0

    return ann_return, sharpe


def analyze_portfolio(returns, regimes, row):

    # Handle different column naming (ETFs vs Assets)
    if "ETFs" in row:
        etfs = row["ETFs"].split(",")
    elif "Assets" in row:
        etfs = row["Assets"].split(",")
    else:
        return {}

# Handle weights safely
    if "Weights" in row:
        weights = np.array([float(x) for x in row["Weights"].split(",")])
    else:
        return {}

    port_returns = compute_portfolio_returns(returns, etfs, weights)

    df = pd.DataFrame({
        "Return": port_returns
    })

    df = df.join(regimes["Regime"], how="inner")

    regime_metrics = {}

    for regime in df["Regime"].unique():

        subset = df[df["Regime"] == regime]["Return"]

        ann_return, sharpe = compute_metrics(subset)

        regime_metrics[f"Return_R{regime}"] = ann_return
        regime_metrics[f"Sharpe_R{regime}"] = sharpe

    sharpes = [v for k, v in regime_metrics.items() if "Sharpe" in k]

    if len(sharpes) > 0:
        avg_sharpe = np.mean(sharpes)
        std_sharpe = np.std(sharpes)
        min_sharpe = np.min(sharpes)
    else:
        avg_sharpe = std_sharpe = min_sharpe = 0

    # 🔥 Robustness score
    robustness = avg_sharpe - std_sharpe

    regime_metrics["Avg_Sharpe"] = avg_sharpe
    regime_metrics["Sharpe_Std"] = std_sharpe
    regime_metrics["Worst_Sharpe"] = min_sharpe
    regime_metrics["Robustness"] = robustness

    return regime_metrics


def main():

    print("Running regime portfolio analysis...")

    returns, regimes = load_data()

    results = []

    for label, path in DATASETS.items():

        df = pd.read_csv(path)

        for _, row in df.iterrows():

            metrics = analyze_portfolio(returns, regimes, row)

            metrics["PortfolioID"] = row["PortfolioID"]
            metrics["Type"] = label

            results.append(metrics)

    out = pd.DataFrame(results)

    out.to_csv(
        "results/regime_portfolio_analysis.csv",
        index=False
    )

    print("\nRegime analysis complete")
    print(out.sort_values("Robustness", ascending=False).head(10))


if __name__ == "__main__":
    main()
