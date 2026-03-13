import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

RETURNS_PATH = "data/processed/etf_returns.csv"


def load_returns():
    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )
    return returns


def correl_distance(corr):
    return np.sqrt((1 - corr) / 2)


def main():

    returns = load_returns()

    corr = returns.corr()

    dist = correl_distance(corr)

    link = linkage(squareform(dist), method="single")

    plt.figure(figsize=(14,8))

    dendrogram(
        link,
        labels=returns.columns,
        leaf_rotation=90,
        leaf_font_size=8
    )

    plt.title("ETF Hierarchical Clustering Dendrogram")

    plt.tight_layout()

    plt.savefig("results/etf_dendrogram.png")

    print("Dendrogram saved")


if __name__ == "__main__":
    main()