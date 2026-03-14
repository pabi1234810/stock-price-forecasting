from data.fetch_data import fetch_stock_data
from models.lstm_model import preprocess, build_lstm, train_model, evaluate_model, plot_results

TICKER   = "TCS.NS"
LOOKBACK = 60

df = fetch_stock_data(TICKER)
X_train, y_train, X_test, y_test, scaler, train_size = preprocess(df, lookback=LOOKBACK)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

model = build_lstm(LOOKBACK)
history = train_model(model, X_train, y_train, epochs=50)
actual, predictions, metrics = evaluate_model(model, X_test, y_test, scaler)
plot_results(df, actual, predictions, train_size, TICKER)
