import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RETURNS_PATH = "data/processed/etf_returns.csv"


def load_returns():

    print("Loading return dataset...")

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def prepare_features(returns):

    features = returns.T

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    return scaled, features.index


def run_pca(data):

    print("Running PCA...")

    pca = PCA(n_components=2)

    components = pca.fit_transform(data)

    return components


def plot_pca(components, tickers):

    plt.figure(figsize=(10,8))

    x = components[:,0]
    y = components[:,1]

    plt.scatter(x,y)

    for i, ticker in enumerate(tickers):
        plt.text(x[i], y[i], ticker)

    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.title("ETF Factor Map (PCA)")

    plt.grid(True)

    plt.savefig("results/pca_factor_map.png")

    print("PCA visualization saved")


def main():

    returns = load_returns()

    data, tickers = prepare_features(returns)

    components = run_pca(data)

    plot_pca(components, tickers)


if __name__ == "__main__":
    main()