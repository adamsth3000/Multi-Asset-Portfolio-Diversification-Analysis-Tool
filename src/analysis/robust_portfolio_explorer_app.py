import streamlit as st
import pandas as pd
import plotly.express as px

DATA_PATH = "results/portfolio_combination_summary.csv"


def main():

    st.set_page_config(layout="wide")

    st.title("Robust Portfolio Structure Explorer")

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        st.warning("No data available.")
        return

    # 🔥 Sidebar Filters
    st.sidebar.header("Filters")

    min_sharpe = st.sidebar.slider(
        "Min Avg Sharpe",
        float(df["Avg_Sharpe"].min()),
        float(df["Avg_Sharpe"].max()),
        float(df["Avg_Sharpe"].quantile(0.5))
    )

    max_sharpe_diff = st.sidebar.slider(
        "Max Sharpe Diff (Stability)",
        float(df["Sharpe_Diff"].min()),
        float(df["Sharpe_Diff"].max()),
        float(df["Sharpe_Diff"].quantile(0.5))
    )

    max_return_diff = st.sidebar.slider(
        "Max Return Diff",
        float(df["Return_Diff"].min()),
        float(df["Return_Diff"].max()),
        float(df["Return_Diff"].quantile(0.5))
    )

    min_frequency = st.sidebar.slider(
        "Min Total Occurrences",
        int((df["Num_Traditional_Portfolios"] + df["Num_Factor_Portfolios"]).min()),
        int((df["Num_Traditional_Portfolios"] + df["Num_Factor_Portfolios"]).max()),
        1
    )

    # 🔥 Derived metric
    df["TotalFrequency"] = (
        df["Num_Traditional_Portfolios"] +
        df["Num_Factor_Portfolios"]
    )

    # 🔥 Apply filters
    filtered = df[
        (df["Avg_Sharpe"] >= min_sharpe) &
        (df["Sharpe_Diff"] <= max_sharpe_diff) &
        (df["Return_Diff"] <= max_return_diff) &
        (df["TotalFrequency"] >= min_frequency)
    ]

    st.subheader("Robust Frontier (Cross-Model Consistency)")

    fig = px.scatter(
        filtered,
        x="Return_Traditional",
        y="Return_Factor",
        color="Avg_Sharpe",
        size="TotalFrequency",
        hover_data=[
            "OverlapAssets",
            "Sharpe_Diff",
            "Return_Diff",
            "Num_Traditional_Portfolios",
            "Num_Factor_Portfolios"
        ],
        title="Traditional vs Factor Return Consistency"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 🔥 Stability vs Performance
    st.subheader("Stability vs Performance")

    fig2 = px.scatter(
        filtered,
        x="Sharpe_Diff",
        y="Avg_Sharpe",
        size="TotalFrequency",
        color="OverlapCount",
        hover_data=["OverlapAssets"],
        title="Stability (Sharpe Diff) vs Performance"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # 🔥 Table view
    st.subheader("Top Robust Portfolio Structures")

    top = filtered.sort_values(
        ["Avg_Sharpe", "TotalFrequency"],
        ascending=False
    ).head(25)

    st.dataframe(top)

    # 🔥 Drill-down
    st.subheader("Inspect Portfolio Structure")

    selected = st.selectbox(
        "Select Combination",
        filtered["OverlapAssets"].unique()
    )

    if selected:

        row = filtered[filtered["OverlapAssets"] == selected].iloc[0]

        st.markdown("### Selected Combination")

        st.write(f"**Assets:** {selected}")
        st.write(f"Avg Sharpe: {row['Avg_Sharpe']:.3f}")
        st.write(f"Sharpe Diff: {row['Sharpe_Diff']:.3f}")
        st.write(f"Return Diff: {row['Return_Diff']:.3f}")
        st.write(f"Frequency: {row['TotalFrequency']}")

        st.markdown("### Source Portfolios")

        st.write("Traditional Portfolio IDs:")
        st.write(row["SimID"])

        st.write("Factor Portfolio IDs:")
        st.write(row["FactorSimID"])


if __name__ == "__main__":
    main()
