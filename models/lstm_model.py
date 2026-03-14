import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import warnings
warnings.filterwarnings("ignore")


def preprocess(df: pd.DataFrame, lookback: int = 60, test_ratio: float = 0.2):
    close = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    train_size = int(len(scaled) * (1 - test_ratio))
    train_data = scaled[:train_size]
    test_data  = scaled[train_size - lookback:]

    def create_sequences(data, lookback):
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, lookback)
    X_test,  y_test  = create_sequences(test_data,  lookback)

    X_train = X_train.reshape(-1, lookback, 1)
    X_test  = X_test.reshape(-1, lookback, 1)

    return X_train, y_train, X_test, y_test, scaler, train_size


def build_lstm(lookback: int = 60) -> Sequential:
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model


def train_model(model, X_train, y_train, epochs=50, batch_size=32,
                model_path="models/lstm_model.keras"):
    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Model saved to {model_path}")
    return history


def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual      = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae  = mean_absolute_error(actual, predictions)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    metrics = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "MAPE (%)": round(mape, 2)}
    print("\n--- LSTM Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return actual, predictions, metrics


def plot_results(df, actual, predictions, train_size, ticker="Stock"):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Full History', color='lightgrey', linewidth=1)
    test_index = df.index[train_size:]
    plt.plot(test_index, actual,      label='Actual Price',    color='blue', linewidth=1.5)
    plt.plot(test_index, predictions, label='LSTM Prediction', color='red',  linewidth=1.5, linestyle='--')
    plt.title(f'{ticker} — LSTM Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/lstm_prediction.png", dpi=150)
    plt.show()
    print("Plot saved to data/lstm_prediction.png")


if __name__ == "__main__":
    from data.fetch_data import fetch_stock_data

    TICKER   = "TCS.NS"
    LOOKBACK = 60

    df = fetch_stock_data(TICKER)
    X_train, y_train, X_test, y_test, scaler, train_size = preprocess(df, lookback=LOOKBACK)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_lstm(LOOKBACK)
    history = train_model(model, X_train, y_train, epochs=50)
    actual, predictions, metrics = evaluate_model(model, X_test, y_test, scaler)
    plot_results(df, actual, predictions, train_size, TICKER)