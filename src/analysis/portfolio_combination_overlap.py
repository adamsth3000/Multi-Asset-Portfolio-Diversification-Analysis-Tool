import pandas as pd
import os

if not os.path.exists("results/factor_portfolio_simulations.csv"):
    raise FileNotFoundError(
        "Run factor_portfolio_simulation first"
    )


SIM_PATH = "results/portfolio_simulations.csv"
FACTOR_SIM_PATH = "results/factor_portfolio_simulations.csv"

TOP_N = 500
SIMILARITY_THRESHOLD = 0.6


def get_top(df, metric="Sharpe"):
    return df.sort_values(metric, ascending=False).head(TOP_N)


def parse_set(row):
    return set(row["ETFs"].split(","))


def jaccard_similarity(a, b):

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0

    return intersection / union


def find_combination_overlaps(sim, factor_sim):

    results = []

    for _, row1 in sim.iterrows():

        set1 = parse_set(row1)

        for _, row2 in factor_sim.iterrows():

            set2 = parse_set(row2)

            sim_score = jaccard_similarity(set1, set2)

            if sim_score >= SIMILARITY_THRESHOLD:

                overlap_assets = sorted(list(set1 & set2))

                results.append({
                    "SimID": row1["PortfolioID"],
                    "FactorSimID": row2["PortfolioID"],
                    "Similarity": sim_score,
                    "OverlapAssets": ",".join(overlap_assets),
                    "OverlapCount": len(overlap_assets),

                    # Traditional metrics
                    "Return_Traditional": row1["AnnualReturn"],
                    "Sharpe_Traditional": row1["Sharpe"],

                    # Factor metrics
                    "Return_Factor": row2["AnnualReturn"],
                    "Sharpe_Factor": row2["Sharpe"]
                })

    return pd.DataFrame(results)


def aggregate_by_combination(df):

    if df.empty:
        return df

    grouped = df.groupby("OverlapAssets")

    summary = grouped.agg({
        "Similarity": "mean",
        "OverlapCount": "mean",

        "Return_Traditional": "mean",
        "Sharpe_Traditional": "mean",

        "Return_Factor": "mean",
        "Sharpe_Factor": "mean",

        # 🔥 Track contributing portfolios
        "SimID": lambda x: list(set(x)),
        "FactorSimID": lambda x: list(set(x))
    }).reset_index()

    # 🔥 Combined metrics
    summary["Avg_Return"] = (
        summary["Return_Traditional"] + summary["Return_Factor"]
    ) / 2

    summary["Avg_Sharpe"] = (
        summary["Sharpe_Traditional"] + summary["Sharpe_Factor"]
    ) / 2

    # 🔥 Stability metrics
    summary["Sharpe_Diff"] = abs(
        summary["Sharpe_Traditional"] - summary["Sharpe_Factor"]
    )

    summary["Return_Diff"] = abs(
        summary["Return_Traditional"] - summary["Return_Factor"]
    )

    # 🔥 Count how many portfolios contributed
    summary["Num_Traditional_Portfolios"] = summary["SimID"].apply(len)
    summary["Num_Factor_Portfolios"] = summary["FactorSimID"].apply(len)

    summary = summary.sort_values(
        ["Avg_Sharpe", "Similarity"],
        ascending=False
    )

    return summary


def main():

    print("Finding combination overlaps with full performance + traceability...")

    sim = pd.read_csv(SIM_PATH)
    factor_sim = pd.read_csv(FACTOR_SIM_PATH)

    top_sim = get_top(sim)
    top_factor = get_top(factor_sim, metric="Score")

    overlaps = find_combination_overlaps(top_sim, top_factor)

    overlaps.to_csv(
        "results/portfolio_combination_overlaps_raw.csv",
        index=False
    )

    summary = aggregate_by_combination(overlaps)

    summary.to_csv(
        "results/portfolio_combination_summary.csv",
        index=False
    )

    print("\nTop performing overlapping combinations:\n")
    print(summary.head(10))


if __name__ == "__main__":
    main()
