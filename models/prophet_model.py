import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    prophet_df = df.reset_index()[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    return prophet_df


def train_prophet(prophet_df: pd.DataFrame):
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    print("Prophet model trained successfully.")
    return model


def forecast(model, periods: int = 180):
    future = model.make_future_dataframe(periods=periods)
    forecast_df = model.predict(future)
    print(f"Forecast generated for {periods} days ahead.")
    return forecast_df


def evaluate_prophet(forecast_df, actual_df):
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    merged = actual_df.merge(forecast_df[['ds', 'yhat']], on='ds', how='inner')
    actual = merged['y'].values
    predicted = merged['yhat'].values

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"\n--- Prophet Metrics ---")
    print(f"  RMSE:     {rmse:.2f}")
    print(f"  MAE:      {mae:.2f}")
    print(f"  MAPE (%): {mape:.2f}")
    return {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE (%)": round(mape, 2)}


def plot_prophet(model, forecast_df, ticker="Stock"):
    fig1 = model.plot(forecast_df)
    plt.title(f'{ticker} — Prophet Forecast')
    plt.tight_layout()
    plt.savefig("data/prophet_forecast.png", dpi=150)
    plt.show()

    fig2 = model.plot_components(forecast_df)
    plt.tight_layout()
    plt.savefig("data/prophet_components.png", dpi=150)
    plt.show()
    print("Plots saved to data/")


if __name__ == "__main__":
    from data.fetch_data import fetch_stock_data

    TICKER = "TCS.NS"
    df = fetch_stock_data(TICKER)
    prophet_df = prepare_prophet_df(df)
    model = train_prophet(prophet_df)
    forecast_df = forecast(model, periods=180)
    metrics = evaluate_prophet(forecast_df, prophet_df)
    plot_prophet(model, forecast_df, TICKER)

    print("\nNext 10 days forecast:")
    future_only = forecast_df[forecast_df['ds'] > prophet_df['ds'].max()]
    print(future_only[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10).to_string(index=False))