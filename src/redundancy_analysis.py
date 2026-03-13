import pandas as pd


CORR_PATH = "data/processed/correlation_matrix.csv"

THRESHOLD = 0.95


def load_corr():

    corr = pd.read_csv(
        CORR_PATH,
        index_col=0
    )

    return corr


def find_redundant_assets(corr):

    redundant_pairs = []

    for i in range(len(corr.columns)):
        for j in range(i):

            if abs(corr.iloc[i, j]) > THRESHOLD:

                asset1 = corr.columns[i]
                asset2 = corr.columns[j]

                redundant_pairs.append((asset1, asset2))

    return redundant_pairs


def main():

    corr = load_corr()

    redundant = find_redundant_assets(corr)

    print("\nHighly correlated ETF pairs:\n")

    for pair in redundant:
        print(pair)


if __name__ == "__main__":
    main()