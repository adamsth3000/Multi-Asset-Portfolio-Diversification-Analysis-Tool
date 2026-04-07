import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.makedirs("results", exist_ok=True)



CORR_PATH = "data/processed/correlation_matrix.csv"


def load_correlation():

    corr = pd.read_csv(
        CORR_PATH,
        index_col=0
    )

    return corr


def plot_heatmap(corr):

    print("Generating correlation heatmap...")

    plt.figure(figsize=(14, 12))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title("ETF Correlation Matrix")

    plt.tight_layout()

    plt.savefig("results/correlation_heatmap.png")

    print("Heatmap saved")


def main():

    corr = load_correlation()

    plot_heatmap(corr)


if __name__ == "__main__":
    main()