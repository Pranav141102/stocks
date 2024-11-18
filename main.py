import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


# Set up the Streamlit app
st.set_page_config(page_title="Real-Time Stock Price Predictor", layout="wide")
st.title("Real-Time Stock Price Predictor App")
st.sidebar.header("Settings")

# User input for stock ID
stock = st.sidebar.text_input("Enter the Stock ID", "")

# Set date range for data fetching
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Load the model
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function to fetch stock data
def fetch_stock_data(stock):
    try:
        data = yf.download(stock, start=start, end=end)
        if data.empty:
            st.error("No data found for the specified stock. Please check the stock ID.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to generate ESG data
def generate_esg_data(length):
    df = pd.DataFrame({
        'Environmental Score': np.random.uniform(50, 100, size=length),
        'Social Score': np.random.uniform(50, 100, size=length),
        'Governance Score': np.random.uniform(50, 100, size=length)
    })
    df['Average Score'] = df[['Environmental Score', 'Social Score', 'Governance Score']].mean(axis=1)
    return df

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, 'orange', label='Moving Average')
    ax.plot(full_data['Close'], 'blue', label='Close Price')
    if extra_data:
        ax.plot(extra_dataset, 'green', label='Extra Data')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Stock Price and Moving Averages")
    ax.legend()
    return fig

# Function to determine currency symbol based on stock exchange
def get_currency_symbol(stock):
    if stock.endswith(".NS") or stock.endswith(".BS"):
        return "₹"  # Indian Rupee
    else:
        return "$"  # Default to Dollar for other exchanges

# Button to fetch data
if st.sidebar.button("Fetch Stock Data"):
    google_data = fetch_stock_data(stock)

    if google_data is not None:
        # Display stock data
        st.subheader("Stock Data Overview")
        st.dataframe(google_data)

        # Generate and display ESG data
        esg_df = generate_esg_data(len(google_data))
        st.subheader("ESG Scores")
        st.dataframe(esg_df)

        # Split the data for training and testing
        splitting_len = int(len(google_data) * 0.7)
        x_test = google_data[['Close']][splitting_len:]

        # Moving Averages
        ma_periods = [100, 200, 250]
        for ma_period in ma_periods:
            google_data[f'MA_for_{ma_period}_days'] = google_data['Close'].rolling(ma_period).mean()
            st.subheader(f'{ma_period}-Day Moving Average')
            st.pyplot(plot_graph((15, 6), google_data[f'MA_for_{ma_period}_days'], google_data))

        # Scale data for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test)

        # Prepare data for the model
        x_data, y_data = [], []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Make predictions
        predictions = model.predict(x_data)

        # Inverse scaling to get actual values
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Prepare data for plotting
        plotting_data = pd.DataFrame({
            'Original Test Data': inv_y_test.flatten(),
            'Predicted Test Data': inv_pre.flatten()
        }, index=google_data.index[splitting_len + 100:])

        # Display original vs predicted values
        st.subheader("Original vs Predicted Values")
        st.dataframe(plotting_data)

        # Plotting the predicted vs original close price
        st.subheader('Close Price Comparison')
        fig = plt.figure(figsize=(15, 6))
        plt.plot(pd.concat([google_data['Close'][:splitting_len + 100], plotting_data], axis=0), label='Original Close Price', color='blue')
        plt.plot(plotting_data['Predicted Test Data'], label='Predicted Close Price', color='orange')
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Comparison of Original and Predicted Stock Prices")
        st.pyplot(fig)

        # Investment Recommendation Section
        latest_close_price = google_data['Close'].iloc[-1]
        ma_100 = google_data[f'MA_for_100_days'].iloc[-1]
        ma_200 = google_data[f'MA_for_200_days'].iloc[-1]

        # Determine currency symbol
        currency_symbol = get_currency_symbol(stock)

        # Convert the values to floats to ensure proper comparison
        latest_close_price = float(latest_close_price)
        ma_100 = float(ma_100)
        ma_200 = float(ma_200)

        # Determine recommendation
        if latest_close_price > ma_100 and latest_close_price > ma_200:
            recommendation = "Buy"
        elif latest_close_price < ma_100 and latest_close_price < ma_200:
            recommendation = "Sell"
        else:
            recommendation = "Hold"
        
        # Display the recommendation
        st.subheader("Investment Recommendation")
        st.markdown(f"<h6>Latest Close Price: {currency_symbol}{latest_close_price:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>100-Day Moving Average: {currency_symbol}{ma_100:.2f}</h6>", unsafe_allow_html=True)
        st.markdown(f"<h6>200-Day Moving Average: {currency_symbol}{ma_200:.2f}</h6>", unsafe_allow_html=True)
        if recommendation == "Buy":
            st.markdown("<h6>Recommendation: <span style='color: green;'>BUY</span></h6>", unsafe_allow_html=True)
        elif recommendation == "Sell":
            st.markdown("<h6>Recommendation: <span style='color: red;'>SELL</span></h6>", unsafe_allow_html=True)
        else:  # Hold
            st.markdown("<h6>Recommendation: <span style='color: #fdc500;'>HOLD</span></h6>", unsafe_allow_html=True)

# Footer or additional info
st.sidebar.subheader("About this App")
st.sidebar.info("We are India’s leading trading app, offering expert guidance across stocks, and more – all at affordable prices. Our platform empowers you with the tools and insights needed for seamless, informed trading.")
st.sidebar.subheader("How It Works")
st.sidebar.info("Enter a stock ticker, and the app will show historical data, predictions, ESG scores, and recommendations.")