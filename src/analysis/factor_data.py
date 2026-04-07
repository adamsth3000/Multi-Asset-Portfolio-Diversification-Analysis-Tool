import pandas as pd
import zipfile
import io
import requests


def get_fama_french_factors():

    print("Fetching Fama-French factors (direct download)...")

    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"

    response = requests.get(url)

    z = zipfile.ZipFile(io.BytesIO(response.content))
    file_name = z.namelist()[0]

    with z.open(file_name) as f:
        df = pd.read_csv(f, skiprows=3)

    # Clean data
    df = df.rename(columns={
        df.columns[0]: "Date",
        "Mkt-RF": "MKT_RF",
        "SMB": "SMB",
        "HML": "HML",
        "RF": "RF"
    })

    # Drop footer rows
    df = df[df["Date"].astype(str).str.isnumeric()]

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date")

    # Convert to decimal
    df = df.astype(float) / 100.0

    return df
