import pandas as pd
from prophet import Prophet
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://neondb_owner:npg_deol3IVuD1Mt@ep-broad-wave-alw7sqw8-pooler.c-3.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

engine = create_engine(
    DATABASE_URL
)
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

forecast_output.to_sql(
    "future_forecast",
    engine,
    if_exists="replace",
    index=False
)
print(
    "✅ Forecast pipeline completed."
)
