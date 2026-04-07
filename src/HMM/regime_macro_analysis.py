import pandas as pd
import numpy as np
from fredapi import Fred

FRED_API_KEY = "2810c7a6501ce73a126adb82ab0dcfa0"


REGIME_PATH = "results/regime_labels.csv"

START_DATE = "2000-01-01"


def load_regimes():

    df = pd.read_csv(
        REGIME_PATH,
        index_col=0,
        parse_dates=True
    )

    return df


def load_macro():

    print("Loading macro data from FRED...")

    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "DGS10": "10Y Treasury",
        "DGS2": "2Y Treasury",
        "CPIAUCSL": "CPI",
        "UNRATE": "Unemployment",
        "FEDFUNDS": "Fed Funds Rate",
        "USREC": "Recession"
    }

    data = {}

    for code, name in series.items():
        try:
            s = fred.get_series(code)
            data[name] = s
        except Exception as e:
            print(f"Failed to load {code}: {e}")

    macro = pd.DataFrame(data)

    macro = macro.ffill()

    return macro


def merge_data(regimes, macro):

    df = regimes.join(macro, how="inner")

    return df


def analyze_regimes(df):

    print("\nAnalyzing macro differences across regimes...\n")

    results = []

    for col in df.columns:

        if col in ["Return", "Volatility", "Drawdown", "Regime"]:
            continue

        grouped = df.groupby("Regime")[col].mean()

        std = df.groupby("Regime")[col].std()

        spread = grouped.max() - grouped.min()

        results.append({
            "Variable": col,
            "Mean_Spread": spread,
            "Std_Avg": std.mean()
        })

        print(f"\n{col}")
        print(grouped)

    return pd.DataFrame(results)


def main():

    regimes = load_regimes()

    macro = load_macro()

    df = merge_data(regimes, macro)

    summary = analyze_regimes(df)
    # Add signal strength metric
    summary["Signal_Strength"] = summary["Mean_Spread"] / summary["Std_Avg"]

    # Sort for readability
    summary = summary.sort_values("Signal_Strength", ascending=False)


    summary.to_csv(
        "results/regime_macro_summary.csv",
        index=False
    )

    print("\nMacro regime analysis saved")


if __name__ == "__main__":
    main()
