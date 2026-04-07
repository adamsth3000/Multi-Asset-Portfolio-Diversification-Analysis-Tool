import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/regime_labels.csv", parse_dates=True, index_col=0)

plt.figure(figsize=(12,6))

for regime in df["Regime"].unique():
    subset = df[df["Regime"] == regime]
    plt.scatter(subset.index, subset["Return"], label=f"Regime {regime}", s=5)

plt.legend()
plt.title("Market Regimes (HMM)")
plt.xlabel("Date")
plt.ylabel("Return")

plt.tight_layout()
plt.savefig("results/regime_visualization.png")

print("Saved regime visualization")
