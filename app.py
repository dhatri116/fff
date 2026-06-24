import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(
    page_title="Stock Forecast using ARIMA",
    layout="wide"
)

st.title("📈 Stock Price Forecast using ARIMA")
st.write("Downloads 5 years of historical stock data and forecasts the price for June 2027.")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
).upper()

if st.button("Generate Forecast"):

    try:

        with st.spinner("Downloading data..."):

            data = yf.download(
                ticker,
                period="5y",
                auto_adjust=True,
                progress=False
            )

        if data.empty:
            st.error("No data found for this ticker.")
            st.stop()

        # Handle yfinance MultiIndex issue
        close_prices = data["Close"]

        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        close_prices = pd.Series(
            close_prices.values,
            index=close_prices.index,
            name="Close"
        )

        st.success(f"Downloaded {len(close_prices)} trading days.")

        st.subheader("Historical Price Chart")

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            close_prices.index,
            close_prices.values
        )

        ax.set_title(f"{ticker} Closing Price (Last 5 Years)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        st.pyplot(fig)

        st.subheader("Recent Data")

        st.dataframe(
            close_prices.tail(10)
        )

        st.subheader("Training ARIMA Model")

        with st.spinner("Fitting ARIMA..."):

            model = ARIMA(
                close_prices,
                order=(5, 1, 0)
            )

            model_fit = model.fit()

        target_date = pd.Timestamp("2027-06-30")

        future_days = len(
            pd.bdate_range(
                start=pd.Timestamp.today(),
                end=target_date
            )
        )

        if future_days <= 0:
            st.error("Target date has already passed.")
            st.stop()

        forecast = model_fit.forecast(
            steps=future_days
        )

        forecast_index = pd.bdate_range(
            start=pd.Timestamp.today(),
            periods=future_days
        )

        forecast_df = pd.DataFrame(
            {
                "Forecast": forecast.values
            },
            index=forecast_index
        )

        june_2027_price = float(
            forecast_df.iloc[-1]["Forecast"]
        )

        st.subheader("Forecast Result")

        st.metric(
            label="Estimated Price on 30-Jun-2027",
            value=f"${june_2027_price:,.2f}"
        )

        st.subheader("Forecast Chart")

        fig2, ax2 = plt.subplots(figsize=(12, 5))

        ax2.plot(
            close_prices.index,
            close_prices.values,
            label="Historical"
        )

        ax2.plot(
            forecast_df.index,
            forecast_df["Forecast"],
            label="Forecast"
        )

        ax2.set_title(
            f"{ticker} Historical + ARIMA Forecast"
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        st.pyplot(fig2)

        st.subheader("Forecast Data")

        st.dataframe(
            forecast_df.tail(20)
        )

    except Exception as e:

        st.error("Error encountered:")
        st.exception(e)
