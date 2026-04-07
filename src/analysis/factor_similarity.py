import pandas as pd
import numpy as np

FACTOR_PATH = "results/factor_exposures.csv"


def load_factor_exposures():
    df = pd.read_csv(FACTOR_PATH, index_col=0)
    return df


def compute_similarity_matrix(factors):

    tickers = factors.index

    matrix = pd.DataFrame(index=tickers, columns=tickers)

    for i in range(len(tickers)):
        for j in range(len(tickers)):

            a = factors.iloc[i][["beta_mkt","beta_smb","beta_hml"]].values
            b = factors.iloc[j][["beta_mkt","beta_smb","beta_hml"]].values

            distance = np.linalg.norm(a - b)

            matrix.iloc[i, j] = distance

    return matrix


def main():

    print("Computing factor similarity matrix...")

    factors = load_factor_exposures()

    similarity = compute_similarity_matrix(factors)

    similarity.to_csv("results/factor_similarity_matrix.csv")

    print("Saved to results/factor_similarity_matrix.csv")


if __name__ == "__main__":
    main()
