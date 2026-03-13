import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RETURNS_PATH = "data/processed/etf_returns.csv"

N_CLUSTERS = 3


def load_returns():

    print("Loading return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def prepare_features(returns):

    print("Preparing clustering features...")

    features = returns.T

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    return scaled, features.index


def run_clustering(data):

    n_assets = data.shape[0]

    clusters = min(10, n_assets - 1)

    print(f"Running KMeans with {clusters} clusters...")

    model = KMeans(
        n_clusters=clusters,
        random_state=42
    )

    labels = model.fit_predict(data)

    return labels


def save_clusters(labels, tickers):

    clusters = pd.DataFrame({
        "ETF": tickers,
        "Cluster": labels
    })

    clusters.to_csv("results/etf_clusters.csv", index=False)

    print("Cluster assignments saved")


def main():

    returns = load_returns()

    data, tickers = prepare_features(returns)

    labels = run_clustering(data)

    save_clusters(labels, tickers)

    print("\nClustering complete")


if __name__ == "__main__":
    main()