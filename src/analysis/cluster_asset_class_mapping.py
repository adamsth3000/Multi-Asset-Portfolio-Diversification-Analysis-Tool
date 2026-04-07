import pandas as pd
import yfinance as yf

CLUSTER_PATH = "results/etf_clusters.csv"


def get_asset_class(ticker):

    try:
        info = yf.Ticker(ticker).info

        return info.get("category") or info.get("sector") or "Unknown"

    except:
        return "Unknown"


def main():

    clusters = pd.read_csv(CLUSTER_PATH)

    results = []

    print("Fetching asset class info...")

    for ticker in clusters["ETF"]:

        asset_class = get_asset_class(ticker)

        cluster = clusters[clusters["ETF"] == ticker]["Cluster"].values[0]

        results.append({
            "ETF": ticker,
            "Cluster": cluster,
            "AssetClass": asset_class
        })

    df = pd.DataFrame(results)

    df.to_csv("results/clusters_itemized_class.csv", index=False)

    print("Saved cluster asset class mapping")


if __name__ == "__main__":
    main()
