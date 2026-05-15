import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

    # =========================
    # PREP DATA
    # =========================

    lstm_df = monthly_sales.reset_index()

    sales_values = lstm_df["Sales"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(sales_values)

    # =========================
    # CREATE SEQUENCES
    # =========================

    X = []
    y = []

    window_size = 3

    for i in range(window_size, len(scaled_data)):

        X.append(scaled_data[i-window_size:i])

        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # =========================
    # TRAIN TEST SPLIT
    # =========================

    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # =========================
    # BUILD MODEL
    # =========================

    model = Sequential()

    model.add(
        LSTM(
            50,
            activation="relu",
            input_shape=(window_size, 1)
        )
    )

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    # =========================
    # TRAIN
    # =========================

    model.fit(
        X_train,
        y_train,
        epochs=50,
        verbose=0
    )

    # =========================
    # PREDICTIONS
    # =========================

    predictions = model.predict(X_test)

    predictions = scaler.inverse_transform(predictions)

    actual = scaler.inverse_transform(y_test)

    # =========================
    # RMSE
    # =========================

    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )

    st.metric(
        "📉 RMSE Accuracy Score",
        f"{rmse:.2f}"
    )

    # =========================
    # ACTUAL vs PREDICTED
    # =========================

    fig_lstm = go.Figure()

    fig_lstm.add_trace(
        go.Scatter(
            y=actual.flatten(),
            mode="lines",
            name="Actual Sales"
        )
    )

    fig_lstm.add_trace(
        go.Scatter(
            y=predictions.flatten(),
            mode="lines",
            name="Predicted Sales"
        )
    )

    fig_lstm.update_layout(
        title="Actual vs Predicted Sales",
        xaxis_title="Time",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_lstm,
        use_container_width=True
    )

    # =========================
    # FUTURE FORECAST
    # =========================

    future_steps = 6

    last_sequence = scaled_data[-window_size:]

    future_predictions = []

    current_sequence = last_sequence.copy()

    for _ in range(future_steps):

        pred = model.predict(
            current_sequence.reshape(1, window_size, 1),
            verbose=0
        )

        future_predictions.append(pred[0][0])

        current_sequence = np.vstack(
            [current_sequence[1:], pred]
        )

    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    # =========================
    # FUTURE DATES
    # =========================

    future_dates = pd.date_range(
        start=monthly_sales.index[-1],
        periods=future_steps + 1,
        freq="ME"
    )[1:]

    # =========================
    # FORECAST CHART
    # =========================

    fig_future = go.Figure()

    fig_future.add_trace(
        go.Scatter(
            x=monthly_sales.index,
            y=monthly_sales.values,
            mode="lines",
            name="Historical Sales"
        )
    )

    fig_future.add_trace(
        go.Scatter(
            x=future_dates,
            y=future_predictions.flatten(),
            mode="lines+markers",
            name="Future Forecast"
        )
    )

    fig_future.update_layout(
        title="LSTM Future Forecast",
        xaxis_title="Date",
        yaxis_title="Sales"
    )

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )

    # =========================
    # AI INSIGHT
    # =========================

    trend = future_predictions[-1][0] - future_predictions[0][0]

    if trend > 0:

        st.success(
            "📈 LSTM predicts upward sales momentum."
        )

    else:

        st.warning(
            "📉 LSTM predicts slowing sales momentum."
        )
