import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/rolling_diversification.csv")

plt.figure(figsize=(12,6))

plt.plot(df["Date"], df["DiversificationRatio"])

plt.title("Rolling Diversification Ratio")

plt.xlabel("Date")
plt.ylabel("Diversification Ratio")

plt.grid(True)

plt.savefig("results/rolling_diversification_plot.png")

print("Plot saved")