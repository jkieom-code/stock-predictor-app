import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Simple Stock Prediction App")
st.write("Enter a stock ticker and click **Download Data** to begin.")

# ----- USER INPUT -----
ticker = st.text_input("Stock Ticker (example: AAPL, TSLA, MSFT):", "AAPL")

# Container to preserve layout
data_container = st.empty()

# ----- DOWNLOAD BUTTON -----
if st.button("📥 Download Data"):
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

        if data.empty:
            st.error("Invalid ticker or no data found.")
        else:
            st.success("Data downloaded!")
            data_container.dataframe(data.tail())

            # ----- FEATURE ENGINEERING -----
            data["Return"] = data["Close"].pct_change()
            data["MA10"] = data["Close"].rolling(10).mean()
            data["MA50"] = data["Close"].rolling(50).mean()
            data = data.dropna()

            # Save for later use (store in session state)
            st.session_state["data"] = data

# ----- PREDICT BUTTON -----
if st.button("🔮 Predict Future Prices"):
    if "data" not in st.session_state:
        st.error("Please download data first.")
    else:
        data = st.session_state["data"]

        # ----- MODEL TRAINING -----
        features = ["Return", "MA10", "MA50"]
        X = data[features]
        y = data["Close"]

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # ----- SHOW PREDICTION CHART -----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test.index, y_test, label="Actual Price", color="blue")
        ax.plot(y_test.index, preds, label="Predicted Price", color="orange")
        ax.legend()
        ax.set_title(f"{ticker} — Actual vs Predicted Prices")

        st.pyplot(fig)

        # ----- NEXT-DAY PREDICTION -----
        next_day_input = X.iloc[-1:].values
        next_pred = model.predict(next_day_input)[0]

        st.subheader("📅 Next-Day Price Prediction")
        st.info(f"**Predicted Closing Price:** ${next_pred:.2f}")
