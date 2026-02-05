import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="ETC Dashboard", layout="wide")
st.title("ETC Revenue Dashboard (FY 2024–25)")

# ---------------- LOAD & PREPARE DATA ----------------
@st.cache_data
def load_data(path):
    df_raw = pd.read_excel(path, header=1)

    base_cols = ["Fee Plaza Name", "State"]
    count_cols = [c for c in df_raw.columns if "Count" in c]
    amount_cols = [c for c in df_raw.columns if "Amount" in c]

    months = [
        "Apr-24", "May-24", "Jun-24", "Jul-24", "Aug-24",
        "Sep-24", "Oct-24", "Nov-24", "Dec-24",
        "Jan-25", "Feb-25", "Mar-25"
    ]

    long_frames = []

    for i, m in enumerate(months):
        temp = df_raw[["Fee Plaza Name", "State", count_cols[i], amount_cols[i]]].copy()
        temp["Month"] = m
        temp.rename(columns={
            count_cols[i]: "Transactions",
            amount_cols[i]: "Revenue"
        }, inplace=True)
        long_frames.append(temp)

    df_long = pd.concat(long_frames, ignore_index=True)

    # Clean numbers
    for col in ["Transactions", "Revenue"]:
        df_long[col] = (
            df_long[col].astype(str)
            .str.replace(",", "", regex=False)
            .replace("nan", np.nan)
        )
        df_long[col] = pd.to_numeric(df_long[col], errors="coerce")

    df_long.dropna(subset=["Fee Plaza Name"], inplace=True)

    df_long["Month"] = pd.Categorical(
        df_long["Month"], categories=months, ordered=True
    )

    return df_long


DATA_PATH = "Monthly-ETC-Data-FY-24-25-11.xlsx"
df_long = load_data(DATA_PATH)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")

# State dropdown
state_options = ["All"] + sorted(df_long["State"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_options)

# Month dropdown
month_options = ["All"] + list(df_long["Month"].cat.categories)
selected_month = st.sidebar.selectbox("Select Month", month_options)

# Top-N slider for plazas
TOP_PLAZAS = 15


# ---------------- APPLY FILTERS ----------------
df_filtered = df_long.copy()

if selected_state != "All":
    df_filtered = df_filtered[df_filtered["State"] == selected_state]

if selected_month != "All":
    df_filtered = df_filtered[df_filtered["Month"] == selected_month]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ---------------- KPI METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue (₹)", f"{df_filtered['Revenue'].sum():,.0f}")
col2.metric("Total Transactions", f"{df_filtered['Transactions'].sum():,.0f}")
col3.metric("Toll Plazas (Filtered)", df_filtered["Fee Plaza Name"].nunique())

# ---------------- STATISTICAL SUMMARY ----------------
st.subheader("Statistical Summary (Monthly Aggregated Data)")

monthly_stats = (
    df_filtered
    .groupby("Month")[["Transactions", "Revenue"]]
    .sum()
    .reset_index()
)

rev = monthly_stats["Revenue"]
txn = monthly_stats["Transactions"]

def compute_stats(series):
    median_val = series.median()
    mad_val = (series - median_val).abs().median()

    return {
        "Mean": series.mean(),
        "Median": median_val,
        "Mode": series.mode()[0] if not series.mode().empty else np.nan,
        "Standard Deviation": series.std(),
        "MAD (Median Absolute Deviation)": mad_val,
        "IQR": series.quantile(0.75) - series.quantile(0.25),
    }


rev_stats = compute_stats(rev)
txn_stats = compute_stats(txn)

summary_df = pd.DataFrame({
    "Statistic": list(rev_stats.keys()),
    "Revenue (₹)": [f"{v:,.2f}" for v in rev_stats.values()],
    "Transactions": [f"{v:,.2f}" for v in txn_stats.values()],
})

st.table(summary_df)

st.divider()

# ---------------- MONTHLY TRENDS ----------------
monthly = (
    df_filtered
    .groupby("Month")[["Transactions", "Revenue"]]
    .sum()
    .reset_index()
)

colL, colR = st.columns(2)

with colL:
    fig1, ax1 = plt.subplots()
    ax1.plot(monthly["Month"], monthly["Transactions"], marker="o")
    ax1.set_title("Monthly Transactions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Transactions")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()
    st.pyplot(fig1)

with colR:
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly["Month"], monthly["Revenue"], marker="o")
    ax2.set_title("Monthly Revenue")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Revenue")
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()
    st.pyplot(fig2)

st.divider()

# ---------------- STATE-WISE REVENUE ----------------
state_rev = (
    df_filtered.groupby("State")["Revenue"]
    .sum()
    .sort_values()
    .reset_index()
)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.barh(state_rev["State"], state_rev["Revenue"])
ax3.set_title("State-wise Total Revenue")
ax3.set_xlabel("Total Revenue")
ax3.set_ylabel("State")
fig3.tight_layout()
st.pyplot(fig3)

# ---------------- TOP N PLAZAS ----------------
top_plazas = (
    df_filtered.groupby("Fee Plaza Name")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(TOP_PLAZAS)
    .reset_index()
)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.barh(top_plazas["Fee Plaza Name"], top_plazas["Revenue"])
ax4.set_title(f"Top {TOP_PLAZAS} Toll Plazas by Revenue")
ax4.set_xlabel("Total Revenue")
ax4.set_ylabel("Fee Plaza Name")
fig4.tight_layout()
st.pyplot(fig4)

# ---------------- DATA TABLE ----------------
with st.expander("Show Filtered Data"):
    st.dataframe(df_filtered)

# ---------------- DENSITY PLOT ----------------
st.divider()
st.subheader("Density Plot: Monthly ETC Revenue")

monthly_dist = (
    df_filtered
    .groupby("Month")["Revenue"]
    .sum()
    .reset_index()
)

fig5, ax5 = plt.subplots(figsize=(6, 4))
monthly_dist["Revenue"].plot(kind="kde", ax=ax5)
ax5.set_xlabel("Monthly Revenue")
ax5.set_ylabel("Density")
ax5.set_title("Density Plot of Monthly ETC Revenue")
fig5.tight_layout()
st.pyplot(fig5)

# ---------------- SCATTER PLOT ----------------
st.divider()
st.subheader("Scatter Plot: Transactions vs Revenue")

scatter_df = (
    df_filtered
    .groupby("Month")[["Transactions", "Revenue"]]
    .sum()
    .reset_index()
)

fig6, ax6 = plt.subplots(figsize=(6, 4))
ax6.scatter(
    scatter_df["Transactions"],
    scatter_df["Revenue"]
)
ax6.set_xlabel("Total Transactions")
ax6.set_ylabel("Total Revenue")
ax6.set_title("Relationship between Transactions and Revenue")
fig6.tight_layout()
st.pyplot(fig6)

corr_val = scatter_df["Transactions"].corr(scatter_df["Revenue"])
st.write(f"**Correlation coefficient:** {corr_val:.3f}")
)
# ---------------- HEATMAP: STATE vs MONTH ----------------
st.subheader("Revenue Heatmap (State vs Month)")

heatmap_df = (
    df_filtered
    .pivot_table(
        values="Revenue",
        index="State",
        columns="Month",
        aggfunc="sum"
    )
)

if heatmap_df.empty:
    st.info("Not enough data to display heatmap.")
else:
    fig_hm, ax_hm = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_df,
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax_hm
    )
    ax_hm.set_title("Revenue Distribution Across States and Months")
    ax_hm.set_xlabel("Month")
    ax_hm.set_ylabel("State")
    st.pyplot(fig_hm)
