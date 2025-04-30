import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf

# === Get Apple stock data ===
df = yf.download('AAPL', start='2020-01-01', end='2024-12-31')

# Optional: Save to CSV
df.to_csv('AAPL_stock.csv')

# Convert index to datetime (in case it's not already)
df.index = pd.to_datetime(df.index)

# === Plot closing price ===
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='AAPL Close Price')
plt.title('Apple Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# === Add moving averages ===
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()
df['SMA200'] = df['Close'].rolling(window=200).mean()

# === Plot with moving averages ===
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='AAPL Close Price')
plt.plot(df['SMA20'], label='SMA20')
plt.plot(df['SMA50'], label='SMA50')
plt.plot(df['SMA200'], label='SMA200')
plt.title('Apple Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# === Generate signals for strategy ===
df['Signal'] = 0
df.loc[df['SMA50'] > df['SMA200'], 'Signal'] = 1
df['Position'] = df['Signal'].shift(1)

# === Calculate returns ===
df['Market_Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Position'] * df['Market_Returns']

# === Calculate cumulative returns ===
df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()

# === Plot cumulative returns ===
plt.figure(figsize=(12,6))
plt.plot(df['Cumulative_Market'], label='Market Returns')
plt.plot(df['Cumulative_Strategy'], label='Strategy Returns (Golden/Death Cross)')
plt.title('Strategy vs Market Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
plt.show()

# === Prepare data for price prediction ===
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

X = np.array(df[['Close']])
y = np.array(df['Target'])

print(f"Number of rows in dataset: {len(df)}")
print(df.head())

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Train Linear Regression ===
model = LinearRegression()
model.fit(X_train, y_train)
df['Predicted_Price'] = model.predict(X)

# === Plot prediction ===
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Actual Price')
plt.plot(df['Predicted_Price'], label='Predicted Price')
plt.title('Actual vs Predicted Apple Stock Price (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# === Evaluate Linear Regression ===
preds = model.predict(X_test)
df.loc[df.index[-len(y_test):], 'Predicted_Price'] = preds

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f'MSE: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# === Define multiple models ===
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}

# === Train and evaluate each model ===
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'MSE': mse, 'R2': r2}

# === Print comparison ===
print("\n\U0001F4CA Model Comparison Results:")
print("{:<20} {:>10} {:>10}".format("Model", "MSE", "R²"))
for name, score in results.items():
    print("{:<20} {:>10.2f} {:>10.2f}".format(name, score['MSE'], score['R2']))

    # === Plot results ===
plt.figure(figsize=(12,6))
plt.plot(df['Cumulative_Market'], label='Market Returns')
plt.plot(df['Cumulative_Strategy'], label='Strategy Returns (Golden/Death Cross)')
plt.title('Strategy vs Market Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Growth of $1')
plt.legend()
plt.grid(True)
plt.show()

# === Export cleaned data for Tableau (with Date) ===
df_to_export = df[['Close', 'SMA20', 'SMA50', 'SMA200', 'Volume', 'Cumulative_Market', 'Cumulative_Strategy']]
df_to_export = df_to_export.copy()
df_to_export['Date'] = df_to_export.index
df_to_export.dropna().to_csv('AAPL_dashboard_data.csv', index=False)
