# 📈 Stock Trend Predictor – TSLA LSTM Model

Built a simple and clean LSTM-based app to forecast Tesla's (TSLA) stock prices using TensorFlow and Streamlit.

The app uses the past 100 days of closing prices to predict future values (default 30 days, but customizable), giving a quick visual idea of short-term trends.

### 🔍 Key Features
- ✅ Trained LSTM model on historical TSLA data
- ✅ Auto-fetches live data using `yfinance`
- ✅ Interactive graph showing actual vs predicted prices
- ✅ RMSE & MAE evaluation metrics for train/test
- ✅ Streamlit UI for quick experimentation

A good hands-on exercise in:
- Time-series preprocessing
- Sequential modeling with LSTMs
- Metric evaluation and result interpretation
- Building user-facing data apps

---

### 🛠 How to Run
1. Clone the repo  
2. Set up a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
3.Run the app
streamlit run streamlit_app.py

###🛠 Tech Stack
Python
TensorFlow
Keras
Scikit-learn
YFinance
Pandas, NumPy, Matplotlib
Streamlit
