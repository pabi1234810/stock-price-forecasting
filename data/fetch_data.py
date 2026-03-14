import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, start: str = "2018-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)

    print(f"Fetched {len(df)} rows from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def save_data(df: pd.DataFrame, ticker: str, path: str = "data/"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{ticker.replace('.', '_')}_stock_data.csv")
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    return filename


if __name__ == "__main__":
    TICKER = "TCS.NS"
    df = fetch_stock_data(TICKER)
    save_data(df, TICKER)
    print(df.tail())