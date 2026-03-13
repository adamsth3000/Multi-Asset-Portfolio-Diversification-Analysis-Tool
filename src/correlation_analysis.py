import pandas as pd


RETURNS_PATH = "data/processed/etf_returns.csv"


def load_returns():

    print("Loading ETF return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def compute_correlation_matrix(returns):

    print("Computing correlation matrix...")

    corr_matrix = returns.corr()

    return corr_matrix


def save_results(corr_matrix):

    corr_matrix.to_csv("data/processed/correlation_matrix.csv")

    print("Correlation matrix saved")


def main():

    returns = load_returns()

    corr_matrix = compute_correlation_matrix(returns)

    save_results(corr_matrix)

    print("\nAnalysis complete")


if __name__ == "__main__":
    main()