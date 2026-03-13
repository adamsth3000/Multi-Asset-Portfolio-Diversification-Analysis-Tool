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
