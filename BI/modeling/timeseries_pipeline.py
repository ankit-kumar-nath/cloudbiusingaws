from prophet import Prophet
import pandas as pd

def train_prophet(df, date_col, value_col, periods=30, freq="D"):
    df2 = df[[date_col, value_col]].dropna().rename(columns={date_col: "ds", value_col: "y"})
    df2["ds"] = pd.to_datetime(df2["ds"])
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df2)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return m, forecast
