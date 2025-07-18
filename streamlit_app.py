import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from io import BytesIO

# Load model and scaler
model = load_model("tsla_lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit App
st.set_page_config(layout="wide")
st.title("📈 TSLA Stock Price Predictor (LSTM)")
st.write("This app uses a trained LSTM model to predict the next N days of Tesla's stock closing prices.")

# Sidebar for inputs
st.sidebar.header("Prediction Settings")
future_days = st.sidebar.slider("Number of future days to predict", min_value=5, max_value=60, value=30, step=5)

# File upload
uploaded_file = st.file_uploader("📤 Upload TSLA CSV with 'Close' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Close' not in df.columns:
        st.error("Uploaded CSV must contain a 'Close' column.")
    else:
        close_prices = df['Close'].values.reshape(-1, 1)

        # Normalize and take last 100 for prediction
        input_data = scaler.transform(close_prices)
        temp_input = input_data[-100:].reshape(1, -1)[0].tolist()
        lst_output = []

        for _ in range(future_days):
            x_input = np.array(temp_input[-100:]).reshape((1, 100, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.append(yhat[0][0])
            lst_output.append(yhat[0][0])

        # Rescale predictions back to original price
        predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

        # Prepare combined data for plot
        past = close_prices[-100:]
        future = predictions
        total = np.concatenate((past, future))

        # Layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"📊 Predicted TSLA Close Prices for Next {future_days} Days")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.arange(100), scaler.inverse_transform(past), label="Last 100 Days", color='blue')
            ax.plot(np.arange(100, 100 + future_days), future, label="Predicted", color='orange')
            ax.set_xlabel("Days")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.set_title("TSLA Price Forecast")
            st.pyplot(fig)

        with col2:
            st.subheader("🔢 Raw Prediction Values")
            pred_df = pd.DataFrame(predictions, columns=["Predicted Close"])
            st.dataframe(pred_df)

            # Download button
            csv_buffer = BytesIO()
            pred_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Download Predictions as CSV", csv_buffer.getvalue(), "predictions.csv", "text/csv")

else:
    st.info("Please upload a CSV file containing TSLA closing prices with a `Close` column.")
