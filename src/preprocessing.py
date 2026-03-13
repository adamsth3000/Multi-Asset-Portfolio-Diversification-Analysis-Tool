import pandas as pd


RAW_DATA_PATH = "data/raw/etf_prices.csv"
OUTPUT_PATH = "data/processed/etf_returns.csv"


def load_price_data(path):
    print("Loading raw price data...")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def remove_assets_with_missing_data(df, threshold=0.9):
    """
    Removes ETFs that have too much missing data.
    threshold = percentage of required observations
    """
    print("Filtering assets with insufficient history...")
    
    valid_counts = df.count()
    required = int(len(df) * threshold)

    valid_assets = valid_counts[valid_counts >= required].index

    return df[valid_assets]


def fill_missing_values(df):
    """
    Forward fill missing values where possible
    """
    print("Filling missing data...")
    
    df = df.ffill()

    return df


def compute_returns(df):
    print("Computing daily returns...")
    
    returns = df.pct_change().dropna()

    return returns


def save_returns(df, path):
    df.to_csv(path)
    print(f"Processed returns saved to {path}")


def main():

    prices = load_price_data(RAW_DATA_PATH)

    prices = remove_assets_with_missing_data(prices)

    prices = fill_missing_values(prices)

    returns = compute_returns(prices)

    save_returns(returns, OUTPUT_PATH)


if __name__ == "__main__":
    main()