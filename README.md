# 📈 Stock Price Forecasting App

A machine learning web application that predicts future stock prices using **Facebook Prophet** time series forecasting. Built with real NSE/NYSE stock data from Yahoo Finance.

🌐 **Live Demo**: [stock-price-forecasting-p5bl55dzc6g9qehimffvfj.streamlit.app](https://stock-price-forecasting-p5bl55dzc6g9qehimffvfj.streamlit.app/)

---

## 🚀 Features

- 📊 Interactive candlestick charts with 20-day and 50-day moving averages
- 🔮 Future price forecasting using Facebook Prophet
- 📉 Confidence interval bands on all forecasts
- 📋 Next 10 days price prediction table
- 📈 Model evaluation metrics — RMSE, MAE, MAPE
- 🔽 Dropdown selector for 15 Indian and US stocks
- ⚡ Built with Streamlit for instant web deployment

---

## 🖥️ Screenshots

> Select a stock → Click Run Forecast → Get predictions instantly

---

## 🧠 Model Used

| Model | Type | Use |
|-------|------|-----|
| Prophet | Time Series (Meta) | Trend + seasonality forecasting |

**Why Prophet?**
- Handles missing data and outliers automatically
- Captures yearly and weekly seasonality
- Gives confidence intervals on predictions
- No manual feature engineering required

---

## 📁 Project Structure
```
stock-forecasting/
│
├── data/
│   ├── __init__.py
│   └── fetch_data.py        # Download stock data via yfinance
│
├── models/
│   ├── __init__.py
│   ├── lstm_model.py        # LSTM deep learning model (local)
│   └── prophet_model.py     # Prophet forecasting model
│
├── app.py                   # Streamlit web application
├── requirements.txt         # Dependencies
├── runtime.txt              # Python version for deployment
└── README.md
```

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/pabi1234810/stock-price-forecasting.git
cd stock-price-forecasting
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📊 Supported Stocks

| Company | Ticker |
|---------|--------|
| TCS | TCS.NS |
| Reliance Industries | RELIANCE.NS |
| Infosys | INFY.NS |
| HDFC Bank | HDFCBANK.NS |
| Wipro | WIPRO.NS |
| ICICI Bank | ICICIBANK.NS |
| State Bank of India | SBIN.NS |
| Adani Enterprises | ADANIENT.NS |
| Bajaj Finance | BAJFINANCE.NS |
| Hindustan Unilever | HINDUNILVR.NS |
| Apple | AAPL |
| Tesla | TSLA |
| Google | GOOGL |
| Microsoft | MSFT |
| Amazon | AMZN |

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error — penalises large errors |
| MAE | Mean Absolute Error — average prediction error |
| MAPE | Mean Absolute Percentage Error — error as % of actual price |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| Streamlit | Web app framework |
| Prophet | Time series forecasting |
| yfinance | Stock data from Yahoo Finance |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |
| Scikit-learn | Evaluation metrics |

---

## 🌐 Deployment

Deployed on **Streamlit Community Cloud** — free hosting for Streamlit apps.

🔗 [https://stock-price-forecasting-p5bl55dzc6g9qehimffvfj.streamlit.app/](https://stock-price-forecasting-p5bl55dzc6g9qehimffvfj.streamlit.app/)

---

## 👤 Author

**Pabitra Chakraborty**
B.E. Mechanical Engineering
Jadavpur University (2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-pabi1234810-black?logo=github)](https://github.com/pabi1234810)