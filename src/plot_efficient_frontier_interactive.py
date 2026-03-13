import pandas as pd
import plotly.express as px

SIM_PATH = "results/portfolio_simulations.csv"

def main():

    df = pd.read_csv(SIM_PATH)

    fig = px.scatter(
        df,
        x="Volatility",
        y="AnnualReturn",
        color="Sharpe",
        hover_data={
            "PortfolioID": True,
            "AnnualReturn": True,
            "Volatility": True,
            "Sharpe": True,
            "ETFs": True
        },
        title="Efficient Frontier (Interactive)"
    )

    fig.write_html("results/efficient_frontier_interactive.html")

    fig.show()


if __name__ == "__main__":
    main()