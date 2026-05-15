import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RetailPulse AI",
    layout="wide"
)

# =========================
# CUSTOM STYLING
# =========================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* KPI cards */
[data-testid="metric-container"] {
    background-color: #1f2937;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 12px;
}

/* Headers */
h1, h2, h3 {
    color: #f9fafb;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Superstore.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])

# =========================
# TITLE
# =========================

st.title("🛍️ RetailPulse")

st.markdown("""
AI-powered retail analytics and forecasting platform
""")

st.markdown("---")

# =========================
# SIDEBAR FILTERS
# =========================

st.sidebar.header("📊 Dashboard Filters")

region = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

segment = st.sidebar.multiselect(
    "Select Segment",
    df["Segment"].unique(),
    default=df["Segment"].unique()
)

# =========================
# FILTER DATA
# =========================

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Segment"].isin(segment))
]

# =========================
# KPI SECTION
# =========================

total_sales = filtered_df["Sales"].sum()

total_profit = filtered_df["Profit"].sum()

total_orders = filtered_df["Order ID"].nunique()

avg_discount = filtered_df["Discount"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")

col2.metric("📈 Total Profit", f"${total_profit:,.0f}")

col3.metric("🛒 Orders", total_orders)

col4.metric("🏷️ Avg Discount", f"{avg_discount:.2f}")

# =========================
# ANALYTICS DATA
# =========================

monthly_sales = filtered_df.resample(
    "M",
    on="Order Date"
)["Sales"].sum()

region_sales = filtered_df.groupby("Region")["Sales"].sum()

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

loss_products = (
    filtered_df.groupby("Sub-Category")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)

best_region = region_sales.idxmax()

best_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs([
    "📈 Sales Analytics",
    "🌍 Regional Insights",
    "🧠 AI Insights"
])

# =========================
# TAB 1
# =========================

with tab1:

    st.subheader("📈 Monthly Sales Trend")

    st.line_chart(monthly_sales)

    st.subheader("🏆 Top Products")

    st.dataframe(top_products)

# =========================
# TAB 2
# =========================

with tab2:

    st.subheader("🌍 Regional Sales")

    st.bar_chart(region_sales)

    st.subheader("📉 Loss-Making Categories")

    st.bar_chart(loss_products)

# =========================
# TAB 3
# =========================

with tab3:

    st.subheader("🧠 AI Business Insights")

    st.success(
        f"""
        📌 Highest sales region: {best_region}

        📌 Best performing category: {best_category}

        📌 Total revenue generated:
        ${total_sales:,.0f}
        """
    )
