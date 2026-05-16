import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("Superstore.csv")

df["Order Date"] = pd.to_datetime(
    df["Order Date"]
)

# =========================
# MONTHLY SALES
# =========================

monthly_sales = df.resample(
    "ME",
    on="Order Date"
)["Sales"].sum().reset_index()

monthly_sales.columns = [
    "ds",
    "y"
]

# =========================
# PROPHET MODEL
# =========================

model = Prophet()

model.fit(monthly_sales)

future = model.make_future_dataframe(
    periods=12,
    freq="ME"
)

forecast = model.predict(future)

# =========================
# SAVE FORECAST
# =========================

forecast_output = forecast[
    [
        "ds",
        "yhat",
        "yhat_lower",
        "yhat_upper"
    ]
]

forecast_output.to_csv(
    "automated_forecast.csv",
    index=False
)

print(
    "✅ Forecast pipeline completed."
)
