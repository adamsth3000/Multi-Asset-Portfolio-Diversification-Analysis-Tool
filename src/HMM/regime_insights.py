import pandas as pd

MACRO_PATH = "results/regime_macro_summary.csv"
PORTFOLIO_PATH = "results/regime_portfolio_analysis.csv"


def load_data():
    macro = pd.read_csv(MACRO_PATH)
    portfolios = pd.read_csv(PORTFOLIO_PATH)
    return macro, portfolios


def identify_strong_macro(macro):

    macro["Signal_Strength"] = macro["Mean_Spread"] / macro["Std_Avg"]

    strong = macro[macro["Signal_Strength"] > 1.5]  # threshold

    print("\nStrong macro signals:\n")
    print(strong.sort_values("Signal_Strength", ascending=False))

    return strong


def portfolio_summary(portfolios):

    print("\nPortfolio robustness by type:\n")

    summary = portfolios.groupby("Type")["Robustness"].agg(
        ["mean", "std", "max"]
    )

    print(summary)

    return summary


def top_portfolios(portfolios):

    print("\nTop robust portfolios:\n")

    top = portfolios.sort_values(
        "Robustness",
        ascending=False
    ).head(10)

    print(top[["PortfolioID", "Type", "Robustness"]])

    return top


def regime_stability(portfolios):

    print("\nRegime stability insights:\n")

    unstable = portfolios[
        portfolios["Sharpe_Std"] > portfolios["Sharpe_Std"].median()
    ]

    print(f"Unstable portfolios: {len(unstable)} / {len(portfolios)}")

    return unstable


def main():

    print("\nGenerating regime insights...\n")

    macro, portfolios = load_data()

    strong_macro = identify_strong_macro(macro)

    summary = portfolio_summary(portfolios)

    top = top_portfolios(portfolios)

    unstable = regime_stability(portfolios)

    print("\n--- FINAL TAKEAWAYS ---\n")

    if len(strong_macro) == 0:
        print("• No strong macro drivers identified → regimes are primarily statistical")
    else:
        print("• Some regimes align with macro conditions (see above)")

    best_type = summary["mean"].idxmax()
    print(f"• Most robust portfolio type: {best_type}")

    print("• Top portfolios show highest consistency across regimes")

    print("\nInsight generation complete\n")


if __name__ == "__main__":
    main()
