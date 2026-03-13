import subprocess
import sys


def run_script(script):

    print("\n----------------------------------")
    print(f"Running {script}")
    print("----------------------------------")

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"Error running {script}")
        sys.exit(1)


def main():

    scripts = [

"src/data_collection.py",
"src/preprocessing.py",

"src/correlation_analysis.py",
"src/plot_correlation_heatmap.py",

"src/cluster_analysis.py",
"src/pca_analysis.py",

"src/diversification_ratio.py",
"src/portfolio_search.py",

"src/rolling_regime_analysis.py",
"src/plot_rolling_diversification.py",

"src/cluster_portfolio.py",
"src/hrp_portfolio.py",

"src/portfolio_metrics.py",
"src/plot_portfolio_growth.py",
"src/plot_portfolio_growth_log.py",

"src/efficient_frontier_simulation.py",
"src/plot_efficient_frontier.py",

"src/plot_dendrogram.py",

"src/monte_carlo_simulation.py"

]

    for script in scripts:
        run_script(script)

    print("\nPipeline complete")


if __name__ == "__main__":
    main()