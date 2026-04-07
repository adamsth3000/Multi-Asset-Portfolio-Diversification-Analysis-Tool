import os
import pandas as pd

# Ensure output directory exists
os.makedirs("results", exist_ok=True)

from src.analysis.factor_data import get_fama_french_factors
from src.analysis.factor_model import compute_factor_exposures



RETURNS_PATH = "data/processed/etf_returns.csv"


def main():

    print("Running factor model step...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    factors = get_fama_french_factors()

    exposures = compute_factor_exposures(returns, factors)

    exposures.to_csv("results/factor_exposures.csv")

    print("Factor model complete")


if __name__ == "__main__":
    main()
