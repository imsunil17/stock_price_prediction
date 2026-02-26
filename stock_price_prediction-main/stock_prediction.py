import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

# Download stock data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google_data = yf.download(stock, start, end)

if google_data is None or google_data.empty:
    st.error("Failed to fetch stock data. Please check the stock ID and try again.")
else:
    # Load the model
    model = load_model(r"C:\Users\jayaragularumugam\Desktop\streamlit\stock_price_prediction\Latest_stock_price_model.keras")

    st.subheader("Stock Data")
    st.write(google_data)

    # Data processing for plotting
    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data.Close[splitting_len:])

    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'Orange')
        plt.plot(full_data.Close, 'b')
        if extra_data:
            plt.plot(extra_dataset)
        return fig

    # Plotting moving averages
    st.subheader('Original Close Price and MA for 250 days')
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 200 days')
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days')
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

    # Scaling data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Predictions on test data
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    plotting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )

    st.subheader("Original values vs Predicted values")
    st.write(plotting_data)

    st.subheader('Original Close Price vs Predicted Close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len + 100], plotting_data], axis=0))
    plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
    st.pyplot(fig)

    # Predict future stock prices
    st.subheader('Predict Future Stock Prices')

    # Date input for prediction
    future_date = st.date_input('Enter the date to predict the stock price:', value=end + timedelta(days=30))

    # Calculate number of days to predict
    last_date = google_data.index[-1]
    days_to_predict = (future_date - last_date.date()).days

    if days_to_predict <= 0:
        st.error("Please select a future date.")
    else:
        # Prepare the latest data
        last_100_days = google_data['Close'][-100:].values
        scaled_last_100_days = scaler.transform(last_100_days.reshape(-1, 1))

        future_predictions = []

        for _ in range(days_to_predict):
            next_prediction = model.predict(scaled_last_100_days[-100:].reshape(1, 100, 1))
            future_predictions.append(next_prediction)
            scaled_last_100_days = np.append(scaled_last_100_days, next_prediction).reshape(-1, 1)

        # Inverse transform the predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create a DataFrame for future predictions
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])

        # Plot future predictions
        st.subheader('Future Stock Price Predictions')
        st.write(future_df)

        fig_future = plt.figure(figsize=(15, 6))
        plt.plot(google_data['Close'], label='Original Data')
        plt.plot(future_df, label='Future Predictions', color='orange')
        plt.legend()
        st.pyplot(fig_future)
