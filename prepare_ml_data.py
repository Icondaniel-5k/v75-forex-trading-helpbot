import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# Load CSV
df = pd.read_csv("v75_historical_data.csv")

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])

# Ensure data is sorted by time
df.sort_values('time', inplace=True)

# Calculate indicators
df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
df['ema20'] = EMAIndicator(df['close'], window=20).ema_indicator()
df['ema50'] = EMAIndicator(df['close'], window=50).ema_indicator()
df['ema200'] = EMAIndicator(df['close'], window=200).ema_indicator()

# Additional Features
df['slope'] = df['close'].diff()
df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
df['price_vs_ema50'] = df['close'] - df['ema50']
df['volatility'] = df['close'].rolling(window=10).std()
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

# Signal labeling with threshold
future_shift = 3
profit_threshold = 0.015  # 1.5%
df['future_return'] = (df['close'].shift(-future_shift) - df['close']) / df['close']
df['signal'] = df['future_return'].apply(lambda x: 1 if x > profit_threshold else 0)

# Drop rows with NaN values
df.dropna(inplace=True)

# Save processed data
df.to_csv("processed_v75_data.csv", index=False)
print(" Indicators and features saved to 'processed_v75_data.csv'")
