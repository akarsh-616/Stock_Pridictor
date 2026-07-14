import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #C9D1D9;
}
h1 {

    color: white;
    text-shadow: 0 0 5px rgba(0, 255, 170, 0.5);
    text-align: center;
    font-family: 'Courier New', Courier, monospace;
}
.stTextInput>div>div>input {
    border: 2px solid #00FFAA;
    border-radius: 8px;
    background-color: #161B22;
    color: white;
}
hr {
    border: 1px solid #00FFAA;
    opacity: 0.2;
}

.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 30px;
    margin-bottom: 10px; /* Space after navbar */
    border: 2px solid #00FFAA;
    border-radius: 25px;
    box-shadow: 0 0 15px rgba(0, 255, 170, 0.6), inset 0 0 10px rgba(0, 255, 170, 0.3);
    background-color: #12171D;
}
.nav-brand {
    font-size: 1.4rem;
    font-weight: bold;
    color: #00FFAA;
    font-family: 'Courier New', Courier, monospace;
    text-shadow: 0 0 10px rgba(0, 255, 170, 0.5);
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-links {
    display: flex;
    gap: 30px;
}
.nav-links a {
    color: #00FFAA;
    text-decoration: none;
    font-family: 'Courier New', Courier, monospace;
    font-size: 1.1rem;
    font-weight: bold;
    transition: text-shadow 0.3s ease;
}
.nav-links a:hover {
    text-shadow: 0 0 20px rgba(0, 255, 170, 0.9);
}
</style>

<!-- Navbar HTML -->
<div class="nav-container">
    <div class="nav-brand">
        📈 Stock Pridictor
    </div>
    <div class="nav-links">
        <a href="#last-month-graph" target="_self">Graph</a>
        <a href="#predicted-value" target="_self">Predictor</a>
        <a href="#about-section" target="_self">About</a>
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---") 
st.title("Stock Pridictor")
st.markdown("---") 
ticker = st.text_input("Enter Stock Ticker (eg. APPL, TSLA, MSFT)", "AAPL")

#appl for apple ,tsla for tesla, msft for microsoft , ^nsei for nifty50

if ticker:
    data = yf.download(ticker, period="1mo", interval="1d")

    if not data.empty:
        st.write(f"Last 1 month data for {ticker}:")
        st.dataframe(data.tail())
        #st.line_chart(dat)

       
        st.markdown('<div id="last-month-graph"></div>', unsafe_allow_html=True)

        close_prices = data["Close"]
        if isinstance(close_prices, pd.DataFrame):
            close_prices = close_prices.iloc[:, 0]

        # Original Graph
        plt.figure(figsize=(10,5))
        plt.plot(data.index, close_prices, label="Close Price", color="#58A6FF")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title(f"{ticker} Stock Price (Last 1 Month)")
        plt.legend()
        
        # Dark theme
        plt.style.use('dark_background')
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        st.pyplot(plt)

      
        st.markdown('<div id="predicted-value"></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Next Day Prediction 🤖")

        # Prepare features for ML (Linear Regression based on time series steps)
        df_ml = pd.DataFrame()
        df_ml['Close'] = close_prices.values
        df_ml['Day'] = np.arange(len(df_ml))

        X = df_ml[['Day']]
        y = df_ml['Close']

      
        model = LinearRegression()
        model.fit(X, y)

        # Predict the next day
        next_day_index = len(df_ml)
        predicted_price = model.predict(pd.DataFrame({'Day': [next_day_index]}))[0]

        # Calculate the next business date
        last_date = data.index[-1]
        next_date = last_date + timedelta(days=1)
        if next_date.weekday() == 5:  # Saturday
            next_date += timedelta(days=2)
        elif next_date.weekday() == 6:  # Sunday
            next_date += timedelta(days=1)

        # Display the DataFrame for the predicted day
        pred_df = pd.DataFrame({
            "Date": [next_date.strftime("%Y-%m-%d")],
            "Predicted Close Price": [round(predicted_price, 2)]
        })
        
        st.write(f"Predicted DataFrame for {ticker} (Next Trading Day):")
        st.dataframe(pred_df)

        # Plot the prediction graph
        plt.figure(figsize=(10,5))
        
        # Ploting old data
        plt.plot(data.index, close_prices, label="Historical Close", marker='.', color="#58A6FF")
        
        # Plot the line pridicted
        plt.plot([last_date, next_date], [close_prices.iloc[-1], predicted_price], color="#00FFAA", linestyle='dashed', label='Prediction Trend')
        
        # Highlight the predicted point
        plt.scatter([next_date], [predicted_price], color="#00FFAA", label="Predicted Close", zorder=5, s=100)
        
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title(f"{ticker} 1-Day ML Forecast")
        plt.legend()
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        st.pyplot(plt)
        
    else:
        st.error("No data found for this ticker. Please check the spelling or try another symbol.")

# Footer
st.markdown('<div id="about-section"></div>', unsafe_allow_html=True)
st.markdown("---")
st.markdown("<div style='text-align: right; font-size: small; color: #C9D1D9;'>Akarsh Singh Creation</div>", unsafe_allow_html=True)
