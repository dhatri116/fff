import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(
    page_title="Stock Price Forecast",
    layout="wide"
)

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
).upper()

if st.button("Generate Forecast"):

    try:
        # Download last 5 years data
        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            st.error("No data found.")
            st.stop()

        close_prices = data["Close"]

        st.subheader("Last 5 Years Price Trend")

        st.line_chart(close_prices)

        st.write(f"Observations: {len(close_prices)}")

        # ARIMA Model
        model = ARIMA(close_prices, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast until June 2027
        today = pd.Timestamp.today()

        target_date = pd.Timestamp("2027-06-30")

        business_days = pd.bdate_range(
            start=today,
            end=target_date
        )

        forecast_steps = len(business_days)

        forecast = model_fit.forecast(
            steps=forecast_steps
        )

        june_2027_price = float(forecast.iloc[-1])

        st.subheader("Forecasted Price")

        st.metric(
            "Estimated Price on 30-Jun-2027",
            f"${june_2027_price:,.2f}"
        )

        # Forecast chart
        forecast_index = business_days

        forecast_df = pd.DataFrame({
            "Forecast": forecast.values
        }, index=forecast_index)

        st.subheader("Forecast Path")

        st.line_chart(forecast_df)

        st.success("Forecast completed successfully.")

    except Exception as e:
        st.error(str(e))
