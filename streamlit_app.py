import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model and scaler
model = load_model("tsla_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit App
st.set_page_config(layout="wide")
st.title("📈 TSLA Stock Price Predictor")
st.write("This app uses a trained LSTM model to predict future closing prices of Tesla stock.")

# Sidebar input
st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Number of future days to predict", min_value=5, max_value=60, value=30, step=5)
lookback_days = st.sidebar.slider("Number of historical days to show", min_value=30, max_value=365, value=90, step=30)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_tsla_data():
    tsla = yf.Ticker("TSLA")
    df = tsla.history(period="2y")  # Get 2 years of data
    return df

try:
    # Fetch TSLA data
    df = fetch_tsla_data()
    close_prices = df['Close'].values.reshape(-1, 1)
    
    if len(close_prices) < 100:
        st.error("Not enough historical data available. Please try again later.")
    else:
        # Scale the data
        scaled_data = scaler.transform(close_prices)

        # Use 80-20 train-test split
        split_idx = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:split_idx]
        test_data = scaled_data[split_idx - 100:]  # include last 100 for continuity

        def create_sequences(data, window=100):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i-window:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        # Predict on train and test
        train_preds = model.predict(X_train, verbose=0)
        test_preds = model.predict(X_test, verbose=0)

        # Inverse scale
        train_preds_inv = scaler.inverse_transform(train_preds)
        y_train_inv = scaler.inverse_transform(y_train)
        test_preds_inv = scaler.inverse_transform(test_preds)
        y_test_inv = scaler.inverse_transform(y_test)

        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_preds_inv))
        train_mae = mean_absolute_error(y_train_inv, train_preds_inv)
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_preds_inv))
        test_mae = mean_absolute_error(y_test_inv, test_preds_inv)

        # Prediction on future
        temp_input = scaled_data[-100:].reshape(1, -1)[0].tolist()
        lst_output = []
        for _ in range(future_days):
            x_input = np.array(temp_input[-100:]).reshape((1, 100, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        # Rescale future predictions
        predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        # Create DataFrame for predictions
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

        # Remove timezones & convert to date
        future_dates = future_dates.tz_localize(None).date  # ✅ This removes timezone & keeps only date

        # Create prediction DataFrame
        pred_df = pd.DataFrame(predictions, index=future_dates, columns=["Predicted Close"])
        pred_df.index.name = "Date"  # Optional: sets index name for clarity


        # Main content layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot section
            st.subheader(f"📊 TESLA Price Prediction")
            
            historical_to_show = df['Close'].iloc[-lookback_days:]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(historical_to_show.index, historical_to_show, label="Historical Prices", color='blue')
            ax.plot(pred_df.index, pred_df["Predicted Close"], label="Predicted Prices", color='orange', linestyle='--')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"Last {lookback_days} Days vs Next {future_days} Days Prediction")
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            # Prediction data section
            st.subheader("🔢 Prediction Data")
            st.dataframe(pred_df.style.format({"Predicted Close": "${:.2f}"}))
            
            st.download_button(
                label="📥 Download Predictions",
                data=pred_df.to_csv().encode('utf-8'),
                file_name="tsla_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
            

        # Model Performance Metrics at the bottom (full width)
        st.markdown("---")
        st.subheader("📏 Model Performance Metrics")
        
        # Create 4 columns for metrics
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Train RMSE", f"${train_rmse:.2f}", 
                     help="Root Mean Squared Error on training data")
        with m2:
            st.metric("Train MAE", f"${train_mae:.2f}", 
                     help="Mean Absolute Error on training data")
        with m3:
            st.metric("Test RMSE", f"${test_rmse:.2f}", 
                     help="Root Mean Squared Error on test data")
        with m4:
            st.metric("Test MAE", f"${test_mae:.2f}", 
                     help="Mean Absolute Error on test data")

except Exception as e:
    st.error(f"Error fetching TSLA data: {str(e)}")
    st.info("Please check your internet connection and try again.")