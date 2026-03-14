import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Stock Price Forecasting", page_icon="📈", layout="wide")
st.title("📈 Stock Price Forecasting App")
st.markdown("Predict future stock prices using **Prophet**")
st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.selectbox("Stock Ticker", options=[
        "TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS",
        "WIPRO.NS", "ICICIBANK.NS", "SBIN.NS", "ADANIENT.NS",
        "BAJFINANCE.NS", "HINDUNILVR.NS", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"
    ])
    start_date    = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    end_date      = st.date_input("End Date",   value=pd.to_datetime("2024-12-31"))
    forecast_days = st.slider("Forecast Days", 30, 365, 180)
    run_btn       = st.button("🚀 Run Forecast", use_container_width=True)
    st.divider()
    st.caption("Data: Yahoo Finance")

@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def run_prophet(df, periods):
    prophet_df = df.reset_index()[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast, prophet_df

def compute_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE (%)": round(mape, 2)}

if run_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        try:
            df = load_data(ticker, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            st.stop()

    if df.empty:
        st.error("No data found. Check ticker symbol.")
        st.stop()

    st.subheader(f"📊 {ticker} — Historical Data")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",  f"₹{df['Close'].iloc[-1]:.2f}")
    c2.metric("52W High",      f"₹{df['Close'].tail(252).max():.2f}")
    c3.metric("52W Low",       f"₹{df['Close'].tail(252).min():.2f}")
    c4.metric("Total Records", len(df))

    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC'
    )])
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-Day MA',
                              line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='20-Day MA',
                              line=dict(color='cyan', width=1.5)))
    fig.update_layout(title=f"{ticker} Price Chart", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔮 Prophet Forecast")
    with st.spinner("Training Prophet model..."):
        model_p, forecast_df, prophet_df = run_prophet(df, forecast_days)

    metrics_p = compute_metrics(
        prophet_df['y'].values,
        forecast_df.loc[forecast_df['ds'].isin(prophet_df['ds']), 'yhat'].values
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", metrics_p["RMSE"])
    m2.metric("MAE",  metrics_p["MAE"])
    m3.metric("MAPE (%)", metrics_p["MAPE (%)"])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'],
                               name='Actual', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'],
                               name='Forecast', line=dict(color='red', dash='dash')))
    fig2.add_trace(go.Scatter(
        x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
        y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'
    ))
    fig2.update_layout(title="Prophet — Actual vs Forecast", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    future_only = forecast_df[forecast_df['ds'] > prophet_df['ds'].max()]
    st.write("**Next 10 Days Forecast:**")
    st.dataframe(
        future_only[['ds','yhat','yhat_lower','yhat_upper']].head(10)
        .rename(columns={'ds':'Date','yhat':'Predicted','yhat_lower':'Lower','yhat_upper':'Upper'})
        .set_index('Date').round(2), use_container_width=True
    )
    st.success("✅ Forecast complete!")

else:
    st.info("👈 Select a stock and click **Run Forecast** to begin.")
    st.markdown("""
    | Company | Ticker |
    |---------|--------|
    | TCS | TCS.NS |
    | Reliance | RELIANCE.NS |
    | Infosys | INFY.NS |
    | HDFC Bank | HDFCBANK.NS |
    | Wipro | WIPRO.NS |
    """)