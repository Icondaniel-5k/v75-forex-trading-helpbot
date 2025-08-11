import pandas as pd

# Load original data
df = pd.read_csv("processed_v75_data.csv")

# Make sure your 'signal' column exists
if 'signal' not in df.columns:
    raise ValueError("Your CSV must have a 'signal' column with 1 for BUY and 0 for SELL.")

# Separate buy and sell trades
buy_trades = df[df['signal'] == 1]
sell_trades = df[df['signal'] == 0]

# Find the smaller group to balance
min_len = min(len(buy_trades), len(sell_trades))

# Sample both to the same size
balanced_df = pd.concat([
    buy_trades.sample(min_len, random_state=42),
    sell_trades.sample(min_len, random_state=42)
])

# Shuffle the combined data
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV
balanced_df.to_csv("balanced_v75_data.csv", index=False)

print(f"✅ Balanced dataset created with {min_len} BUY and {min_len} SELL trades.")
