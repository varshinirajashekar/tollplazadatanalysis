import streamlit as st
import pandas as pd
import numpy as np
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

# ---------------- SIDEBAR DROPDOWNS ----------------
st.sidebar.header("Filters")

state_options = ["All"] + sorted(df_long["State"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State", state_options)

if selected_state == "All":
    df_state = df_long.copy()
else:
    df_state = df_long[df_long["State"] == selected_state]

plaza_options = ["All"] + sorted(df_state["Fee Plaza Name"].unique().tolist())
selected_plaza = st.sidebar.selectbox("Select Fee Plaza", plaza_options)

month_options = ["All"] + list(df_long["Month"].cat.categories)
selected_month = st.sidebar.selectbox("Select Month", month_options)

# ---------------- APPLY FILTERS ----------------
df_filtered = df_long.copy()

if selected_state != "All":
    df_filtered = df_filtered[df_filtered["State"] == selected_state]

if selected_plaza != "All":
    df_filtered = df_filtered[df_filtered["Fee Plaza Name"] == selected_plaza]

if selected_month != "All":
    df_filtered = df_filtered[df_filtered["Month"] == selected_month]

if df_filtered.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ---------------- KPI METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue (₹)", f"{df_filtered['Revenue'].sum():,.0f}")
col2.metric("Total Transactions", f"{df_filtered['Transactions'].sum():,.0f}")
col3.metric("Toll Plazas", df_filtered["Fee Plaza Name"].nunique())

st.divider()

# ---------------- MONTHLY TRENDS ----------------
monthly = df_filtered.groupby("Month")[["Transactions", "Revenue"]].sum().reset_index()

colL, colR = st.columns(2)

with colL:
    fig1, ax1 = plt.subplots()
    ax1.plot(monthly["Month"], monthly["Transactions"], marker="o")
    ax1.set_title("Monthly Transactions")
    ax1.tick_params(axis="x", rotation=45)
    st.pyplot(fig1)

with colR:
    fig2, ax2 = plt.subplots()
    ax2.plot(monthly["Month"], monthly["Revenue"], marker="o")
    ax2.set_title("Monthly Revenue")
    ax2.tick_params(axis="x", rotation=45)
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
st.pyplot(fig3)

# ---------------- TOP 10 PLAZAS ----------------
top_plazas = (
    df_filtered.groupby("Fee Plaza Name")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.barh(top_plazas["Fee Plaza Name"], top_plazas["Revenue"])
ax4.set_title("Top 10 Toll Plazas by Revenue")
st.pyplot(fig4)

# ---------------- DATA TABLE ----------------
with st.expander("Show Filtered Data"):
    st.dataframe(df_filtered)
