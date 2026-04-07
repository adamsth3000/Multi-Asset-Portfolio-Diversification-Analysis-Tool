import pandas as pd
import statsmodels.api as sm


def compute_factor_exposures(returns, factors):

    print("Computing factor exposures...")

    results = {}

    for asset in returns.columns:

        df = pd.DataFrame({
            "asset": returns[asset]
        }).dropna()

        df = df.join(factors, how="inner")

        if len(df) < 100:
            continue

        y = df["asset"] - df["RF"]

        X = df[["MKT_RF", "SMB", "HML"]]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        results[asset] = {
            "alpha": model.params["const"],
            "beta_mkt": model.params["MKT_RF"],
            "beta_smb": model.params["SMB"],
            "beta_hml": model.params["HML"],
            "r_squared": model.rsquared
        }

    exposures = pd.DataFrame(results).T

    exposures.to_csv("results/factor_exposures.csv")

    print("Factor exposures saved")

    return exposures
