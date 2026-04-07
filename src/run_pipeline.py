import subprocess
import sys

import os

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


def run_script(script):

    print("\n----------------------------------")
    print(f"Running {script}")
    print("----------------------------------")

    if script.startswith("-m"):
        cmd = [sys.executable] + script.split()
    else:
        cmd = [sys.executable, script]

    result = subprocess.run(cmd)

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

"-m src.analysis.run_factor_model",

"-m src.analysis.factor_similarity",
"-m src.analysis.factor_cluster_analysis",

"src/diversification_ratio.py",
"src/portfolio_search.py",

"src/rolling_regime_analysis.py",
"src/plot_rolling_diversification.py",

"src/cluster_portfolio.py",
"-m src.analysis.factor_cluster_portfolio",
"-m src.analysis.factor_portfolio_simulation",
"-m src.analysis.portfolio_combination_overlap",
"-m src.analysis.hrp_factor_portfolio",
"-m src.analysis.plot_efficient_frontier_factored",
"-m src.analysis.cluster_asset_class_mapping",
"-m src.analysis.portfolio_metrics_factored",

"src/hrp_portfolio.py",

"src/portfolio_metrics.py",
"src/plot_portfolio_growth.py",
"src/plot_portfolio_growth_log.py",

"src/efficient_frontier_simulation.py",
"src/plot_efficient_frontier.py",

"src/plot_dendrogram.py",

"src/monte_carlo_simulation.py"

# --- FACTOR ANALYSIS ---
"src/analysis/run_factor_model.py",
"src/analysis/factor_portfolio_simulation.py",

# --- COMBINATION ANALYSIS ---
"src/analysis/portfolio_combination_overlap.py",
"src/analysis/combination_portfolio_simulation.py",
"src/analysis/robust_portfolio_builder.py",

# --- HMM REGIME ANALYSIS ---
"src/HMM/regime_detection_hmm.py",
"src/HMM/regime_portfolio_analysis.py",
"src/HMM/regime_macro_analysis.py",
"src/HMM/regime_insights.py",

]

    for script in scripts:
        run_script(script)

    print("\nPipeline complete")


if __name__ == "__main__":
    main()