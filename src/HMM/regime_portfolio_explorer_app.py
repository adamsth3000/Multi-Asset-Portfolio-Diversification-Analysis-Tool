import streamlit as st
import pandas as pd
import plotly.express as px

DATA_PATH = "results/regime_portfolio_analysis.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def main():

    st.title("Portfolio Regime Robustness Explorer")

    df = load_data()

    st.sidebar.header("Filters")

    portfolio_type = st.sidebar.multiselect(
        "Portfolio Type",
        df["Type"].unique(),
        default=df["Type"].unique()
    )

    min_robustness = st.sidebar.slider(
        "Minimum Robustness",
        float(df["Robustness"].min()),
        float(df["Robustness"].max()),
        float(df["Robustness"].min())
    )

    filtered = df[
        (df["Type"].isin(portfolio_type)) &
        (df["Robustness"] >= min_robustness)
    ]

    st.subheader("Top Robust Portfolios")

    top = filtered.sort_values("Robustness", ascending=False).head(20)

    st.dataframe(top)

    st.subheader("Robustness Distribution")

    fig = px.histogram(
        filtered,
        x="Robustness",
        color="Type",
        barmode="overlay",
        title="Robustness Distribution by Portfolio Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sharpe vs Stability")

    fig2 = px.scatter(
        filtered,
        x="Sharpe_Std",
        y="Avg_Sharpe",
        color="Type",
        hover_data=["PortfolioID"],
        title="Sharpe vs Stability"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Inspect Portfolio")

    selected_id = st.selectbox(
        "Select Portfolio ID",
        filtered["PortfolioID"]
    )

    selected = filtered[filtered["PortfolioID"] == selected_id].iloc[0]

    st.write("### Portfolio Details")
    st.write(selected)

    st.subheader("Regime Performance")

    regime_cols = [col for col in df.columns if "Sharpe_R" in col]

    regime_data = {
        col: selected[col]
        for col in regime_cols
    }

    regime_df = pd.DataFrame(
        list(regime_data.items()),
        columns=["Regime", "Sharpe"]
    )

    fig3 = px.bar(
        regime_df,
        x="Regime",
        y="Sharpe",
        title="Sharpe by Regime"
    )

    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
