# Multi Asset Portfolio Diversification Analysis Tool

## Overview

This project is a quantitative research framework for analyzing diversification across a large universe of ETFs and mutual funds.  
It builds a full data pipeline that downloads historical asset prices, analyzes asset relationships, and constructs diversified portfolios using modern quantitative finance techniques.

The system evaluates how different asset classes interact across time and identifies portfolio allocations that maximize diversification.

---

## Objectives

The goal of this project is to answer:

- Which asset classes provide true diversification?
- How do asset relationships change across market regimes?
- What combinations of assets produce the most stable portfolios?

---

## Data Pipeline

The pipeline automatically performs the following steps:

1. Download historical ETF and mutual fund data from Yahoo Finance
2. Clean and preprocess time-series data
3. Generate daily return datasets
4. Compute correlation matrices across assets
5. Identify redundant assets
6. Cluster assets based on return behavior
7. Perform PCA factor analysis
8. Evaluate diversification metrics
9. Construct diversified portfolios using Hierarchical Risk Parity
10. Run Monte Carlo simulations to evaluate portfolio risk

All steps can be executed with a single command:


---

## Methodology

### Correlation Analysis

Measures relationships between assets to identify diversification opportunities.

### Machine Learning Clustering

Uses unsupervised learning to group assets that behave similarly.

### Principal Component Analysis (PCA)

Identifies the underlying factors driving portfolio returns.

### Diversification Ratio

Quantifies how much diversification reduces portfolio risk.

### Random Portfolio Search

Generates thousands of portfolios to identify highly diversified asset combinations.

### Hierarchical Risk Parity (HRP)

Constructs portfolios by allocating risk across hierarchical asset clusters.

### Monte Carlo Simulation

Simulates thousands of potential future portfolio paths using both:

- Bootstrap sampling of historical returns
- Parametric simulations assuming statistical distributions

---

## Visualizations

The system generates multiple visual outputs including:

- Correlation heatmaps
- Hierarchical clustering dendrograms
- PCA factor maps
- Monte Carlo simulation charts
- Rolling diversification regime plots

---

## Technologies Used

Python  
pandas  
numpy  
scikit-learn  
SciPy  
matplotlib  
seaborn  

---

## Key Takeaways

This framework demonstrates how modern quantitative techniques can be used to:

- analyze diversification across many assets
- identify structural relationships in financial markets
- construct robust multi-asset portfolios

---

## Future Improvements

Potential extensions include:

- regime detection using Hidden Markov Models
- factor-based portfolio optimization
- dynamic asset allocation
- integration with macroeconomic indicators


### Example Outputs
- Correlation heatmap

- ETF clustering dendrogram

- Principal Component Analysis factor map

- Monte Carlo simulation chart (Parametric & Bootstrap)

- Rolling Diversification Plot

- ETF/MUTUAL FUND List Itemized by Cluster

- Cluster Diversified Portfolio (10 Clusters)

- Heirarchical Risk Parity Portfolio

- Rolling Diversification by Time Series

# INSTRUCTIONS

# **Command to Activate the Virtual Environment (PowerShell)**

From your project root folder run:

.\venv\Scripts\Activate.ps1

After running it, your prompt should change to:

"(venv) PS C:..."

## Required Python Packages

### Core data libraries

pandas

numpy

### Financial data

yfinance

### Machine learning

scikit-learn

### Scientific computing

scipy

Visualization

matplotlib

seaborn

plotly

### Interactive dashboard

streamlit

## One Command to Install Everything

After activating your virtual environment, run:

python -m pip install pandas numpy yfinance scikit-learn scipy matplotlib seaborn plotly streamlit

This installs everything your project uses.

### Running the Analysis Pipeline

Run the full research pipeline from the project root directory:

python src\run_pipeline.py

This will automatically execute the following steps:

- Download financial data

- Clean and preprocess time-series datasets

- Perform correlation and clustering analysis

- Generate portfolio construction strategies

- Run portfolio simulations

- Generate efficient frontier visualizations

- Produce performance charts and metrics

- Run Monte Carlo simulations

All outputs will be saved to the results/ directory.

### Launch the Interactive Portfolio Explorer

## Start the interactive dashboard:

streamlit run src\portfolio_explorer_app.py

This opens a browser application where you can:

- explore the efficient frontier

- inspect portfolio allocations

- filter portfolios by Sharpe ratio or volatility

- compare multiple portfolios

### Generate a Portfolio Comparison Report

To compare specific portfolios against the benchmark strategies (HRP and Cluster):

python src\portfolio_comparison_tool.py

You will then be prompted to enter portfolio IDs one at a time.

Example interaction:

Portfolio ID: 14
Portfolio ID: 88
Portfolio ID: 420
Portfolio ID:

Press Enter on a blank line to run the analysis.

The script will generate:

results\portfolio_growth_comparison_custom.png

- showing historical growth comparisons between the selected portfolios and the benchmark strategies.

### Generate an Interactive Efficient Frontier

To explore the efficient frontier in a browser:

python src\plot_efficient_frontier_interactive.py

This creates an interactive HTML file:

results\efficient_frontier_interactive.html

Open this file in your browser to interact with the frontier visualization.

### Optional: Run Individual Analysis Scripts

If needed, individual scripts can be run directly:

## Download market data:

python src\data_collection.py

## Preprocess historical price data:

python src\preprocessing.py

## Run clustering analysis:

python src\cluster_analysis.py

## Generate efficient frontier simulation:

python src\efficient_frontier_simulation.py

## Run Monte Carlo simulation:

python src\monte_carlo_simulation.py
Project Structure
src/
│
├── run_pipeline.py
├── portfolio_explorer_app.py
├── portfolio_comparison_tool.py
├── plot_efficient_frontier_interactive.py

Pipeline scripts generate reproducible research outputs, while the explorer and comparison tools provide interactive analysis capabilities.
