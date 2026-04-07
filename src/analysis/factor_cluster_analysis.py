import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FACTOR_PATH = "results/factor_exposures.csv"


def load_factors():
    df = pd.read_csv(FACTOR_PATH, index_col=0)
    return df


def prepare_features(factors):

    features = factors[["beta_mkt","beta_smb","beta_hml"]]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, features.index


def run_clustering(data):

    n_clusters = 6

    model = KMeans(n_clusters=n_clusters, random_state=42)

    labels = model.fit_predict(data)

    return labels


def save_clusters(labels, tickers):

    df = pd.DataFrame({
        "ETF": tickers,
        "FactorCluster": labels
    })

    df.to_csv("results/factor_clusters.csv", index=False)

    print("Factor clusters saved")


def main():

    print("Running factor-based clustering...")

    factors = load_factors()

    data, tickers = prepare_features(factors)

    labels = run_clustering(data)

    save_clusters(labels, tickers)


if __name__ == "__main__":
    main()
