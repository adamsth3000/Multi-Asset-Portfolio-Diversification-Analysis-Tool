import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

RETURNS_PATH = "data/processed/etf_returns.csv"
OUTPUT_PATH = "results/regime_labels.csv"

N_REGIMES = 4


def load_returns():

    returns = pd.read_csv(
        RETURNS_PATH,
        index_col=0,
        parse_dates=True
    )

    return returns


def build_market_series(returns):
    """
    Use equal-weighted market proxy
    """
    market = returns.mean(axis=1)
    return market


def compute_features(market_returns):

    df = pd.DataFrame(index=market_returns.index)

    # Raw returns
    df["Return"] = market_returns

    # Rolling volatility (21-day)
    df["Volatility"] = market_returns.rolling(21).std()

    # Drawdown
    cumulative = (1 + market_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak

    df["Drawdown"] = drawdown

    df = df.dropna()

    return df


def fit_hmm(features):

    from sklearn.preprocessing import StandardScaler

    # ✅ Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # ✅ Convert to numpy array
    X = np.array(X)

    model = GaussianHMM(
        n_components=N_REGIMES,
        covariance_type="diag",   # 🔥 key fix
        n_iter=1000,
        random_state=42
    )

    model.fit(X)

    regimes = model.predict(X)

    return regimes


def main():

    print("Running HMM regime detection...")

    returns = load_returns()

    market = build_market_series(returns)

    features = compute_features(market)

    regimes = fit_hmm(features)

    features["Regime"] = regimes

    features.to_csv(OUTPUT_PATH)

    print("\nRegime detection complete")
    print(f"Saved to {OUTPUT_PATH}")

    print("\nRegime distribution:")
    print(features["Regime"].value_counts())


if __name__ == "__main__":
    main()
