import yfinance as yf
import pandas as pd
from datetime import datetime


# Master ticker list
RAW_TICKERS = [
"VFIAX","VTSAX","FXAIX","VTI","VVIAX","VSIAX","VGSLX","VTIAX","VTRIX","VFSAX",
"VEMAX","VBIRX","FTIHX","FZROX","IVV","IJS","SLYV","USRT","VXUS","EFA",
"IEMG","BSV","VCSH","ITOT","SCHB","SPY","VOO","VBR","VBK","IJT",
"ISCG","SCHH","REET","SCHF","ACWX","AVUV","IVLU","AVIV","VWO","SCHE",
"SWSBX","SUB","SPHB","JUEAX","VRTIX","MASKX","SWSSX","FSSNX","IWM","VTWO",
"DLS","DXJ","DWX","EWG","EWU","EWW","FEZ","FIREX","FIVLX","FLPSX",
"FSCOX","FSPSX","FELV","FEMKX","FHKFX","FLCOX","FNCMX","FPADX","FSRNX","REIT",
"HEDJ","IDV","IFGL","KRE","MACSX","MPACX","MDY","NOBLE","QQQ","RSP",
"SCHA","SCHV","SCHX","SCZ","SDY","SWPPX","VDE","VEA","VFH","VSS",
"VTV","VGENX","VNQ","VYM","YACKX","TGINX"
]

# Automatically remove duplicates and sort
TICKERS = sorted(list(set(RAW_TICKERS)))

print(f"Loaded {len(TICKERS)} unique tickers")
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")


def download_ticker(ticker):

    try:
        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            print(f"{ticker} returned no data")
            return None

        close_prices = data["Close"]
        close_prices.name = ticker

        return close_prices

    except Exception as e:
        print(f"{ticker} failed: {e}")
        return None


def main():

    print("Starting ETF data download")

    price_series = []
    failed_tickers = []

    for ticker in TICKERS:

        result = download_ticker(ticker)

        if result is not None:
            price_series.append(result)
        else:
            failed_tickers.append(ticker)

    if not price_series:
        print("No data downloaded. Check ticker list or connection.")
        return

    prices = pd.concat(price_series, axis=1)

    prices.to_csv("data/raw/etf_prices.csv")

    print("\nDownload complete")

    print(f"\nSuccessful tickers: {len(price_series)}")

    if failed_tickers:
        print("\nFailed tickers:")
        print(failed_tickers)


if __name__ == "__main__":
    main()