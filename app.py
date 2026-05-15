from sqlalchemy import create_engine
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from fpdf import FPDF

DATABASE_URL = "postgresql://neondb_owner:npg_deol3IVuD1Mt@ep-broad-wave-alw7sqw8-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

engine = create_engine(DATABASE_URL)
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
# HEADER
# =========================

st.title("🛍️ RetailPulse AI")

st.markdown("""
AI-powered retail analytics and forecasting platform
""")

st.caption(
    "Built with AI Forecasting, Interactive BI Analytics, and Enterprise Reporting"
)

st.markdown("---")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Superstore.csv")

df["Order Date"] = pd.to_datetime(df["Order Date"])

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
    "ME",
    on="Order Date"
)["Sales"].sum().reset_index()

region_sales = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

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
    .reset_index()
)

best_region = region_sales.loc[
    region_sales["Sales"].idxmax(),
    "Region"
]

best_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

sales_by_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Sales Analytics",
    "🌍 Regional Insights",
    "🧠 AI Insights",
    "🔮 Prophet Forecast",
    "🧠 LSTM Forecast"
])

# =========================
# TAB 1 — SALES ANALYTICS
# =========================

with tab1:

    st.subheader("📈 Monthly Sales Trend")

    fig_sales = px.line(
        monthly_sales,
        x="Order Date",
        y="Sales",
        title="Monthly Sales Trend",
        markers=True
    )

    st.plotly_chart(fig_sales, use_container_width=True)

    st.subheader("🥧 Revenue Distribution")

    fig_pie = px.pie(
        sales_by_category,
        names="Category",
        values="Sales",
        title="Revenue Distribution"
    )

    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("🏆 Top Products")

    st.dataframe(top_products)

# =========================
# TAB 2 — REGIONAL INSIGHTS
# =========================

with tab2:

    st.subheader("🌍 Regional Sales")

    fig_region = px.bar(
        region_sales,
        x="Region",
        y="Sales",
        color="Region",
        title="Regional Sales Performance"
    )

    st.plotly_chart(fig_region, use_container_width=True)

    st.subheader("📉 Loss-Making Categories")

    fig_loss = px.bar(
        loss_products,
        x="Sub-Category",
        y="Profit",
        color="Profit",
        title="Loss-Making Categories"
    )

    st.plotly_chart(fig_loss, use_container_width=True)

# =========================
# TAB 3 — AI INSIGHTS
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

# =========================
# TAB 4 — FORECASTING
# =========================

with tab4:

    st.subheader("🔮 Prophet Sales Forecast")

    prophet_df = monthly_sales.rename(
        columns={
            "Order Date": "ds",
            "Sales": "y"
        }
    )

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=12,
        freq="ME"
    )

    forecast = model.predict(future)

    fig_forecast = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="12-Month Sales Forecast"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader("📊 Forecast Data")

    st.dataframe(
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(12)
    )
# =========================
# 🧠 LSTM TAB
# =========================

with tab5:

    st.subheader("🧠 LSTM Deep Learning Forecast")

    # LOAD FILES
    future_df = pd.read_csv("future_forecast.csv")
future_df.to_sql(
    "future_forecasts",
    engine,
    if_exists="replace",
    index=False
)
    actual_df = pd.read_csv("actual_vs_predicted.csv")

    # =========================
    # ACTUAL vs PREDICTED
    # =========================

    fig_actual = go.Figure()

    fig_actual.add_trace(
        go.Scatter(
            y=actual_df["Actual"],
            mode="lines",
            name="Actual Sales"
        )
    )

    fig_actual.add_trace(
        go.Scatter(
            y=actual_df["Predicted"],
            mode="lines",
            name="Predicted Sales"
        )
    )

    fig_actual.update_layout(
        title="Actual vs Predicted Sales",
        xaxis_title="Time",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_actual,
        use_container_width=True
    )

    # =========================
    # FUTURE FORECAST
    # =========================

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=future_df["Date"],
            y=future_df["Forecast"],
            mode="lines+markers",
            name="Future Forecast"
        )
    )

    fig_future.update_layout(
        title="LSTM Future Forecast",
        xaxis_title="Date",
        yaxis_title="Forecasted Sales"
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )

    st.success(
        "📈 Deep learning forecast engine active."
    )
# =========================
# 📥 EXECUTIVE REPORT
# =========================

st.subheader("📥 Download Executive Report")

if st.button("Generate AI Business Report"):

    report = FPDF()

    report.add_page()

    report.set_font("Arial", "B", 16)

    report.cell(200, 10, "RetailPulse AI Executive Report", ln=True)

    report.ln(10)

    report.set_font("Arial", size=12)

    report.multi_cell(
        0,
        10,
        f"""
Total Sales: ${total_sales:,.0f}

Total Profit: ${total_profit:,.0f}

Total Orders: {total_orders}

Best Region: {best_region}

Best Category: {best_category}

Average Discount: {avg_discount:.2f}

AI Forecast Insight:
RetailPulse predicts continued seasonal fluctuations with
strong Q4 performance patterns.

Strategic Recommendation:
Focus inventory allocation toward high-performing categories
and optimize discount strategies in weaker regions.
"""
    )

    report.output("RetailPulse_Report.pdf")

    with open("RetailPulse_Report.pdf", "rb") as file:

        st.download_button(
            label="📥 Download Report",
            data=file,
            file_name="RetailPulse_Report.pdf",
            mime="application/pdf"
        )
