import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import joblib

MODEL_FILE = "ml_model.pkl"
DATA_FILE = "processed_v75_data.csv"

SYMBOL = "Volatility 75 Index"
LOT_SIZE = 0.01
SLIPPAGE = 5
STOP_LOSS = 1000
TAKE_PROFIT = 1000
MAGIC_NUMBER = 100234
COOLDOWN_MINUTES = 5  # Optional cooldown to space out trades

last_trade_time = None  # Track last trade time


def initialize():
    if not mt5.initialize():
        print("initialize() failed", mt5.last_error())
        quit()

def shutdown():
    mt5.shutdown()

def get_signal():
    df = pd.read_csv(DATA_FILE)
    latest = df.iloc[-1]

    rsi = latest['rsi']
    ema20 = latest['ema20']
    ema50 = latest['ema50']
    ema200 = latest['ema200']
    slope = latest['slope']
    close = latest['close']

    features = np.array([[rsi, ema20, ema50, ema200, slope, close]])
    model = joblib.load(MODEL_FILE)
    prediction = model.predict(features)
    return prediction[0]

def place_trade(signal):
    global last_trade_time

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("Failed to retrieve tick data.")
        return

    price = tick.ask if signal == 1 else tick.bid
    if price is None or price == 0:
        print("Invalid price data.")
        return

    sl = price - STOP_LOSS if signal == 1 else price + STOP_LOSS
    tp = price + TAKE_PROFIT if signal == 1 else price - TAKE_PROFIT

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": SLIPPAGE,
        "magic": MAGIC_NUMBER,
        "comment": "ML Bot Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.order_send(request)
    print("Trade Result:", result)
    last_trade_time = datetime.now()

def check_existing_positions():
    positions = mt5.positions_get(symbol=SYMBOL)
    return len(positions) > 0

def train_model():
    print("Training ML model using:", DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)
    X = df[['rsi', 'ema20', 'ema50', 'ema200', 'slope', 'close']]
    y = df['signal']
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Model training completed and saved to", MODEL_FILE)

initialize()
train_model()
try:
    while True:
        if not check_existing_positions():
            signal = get_signal()
            print(f"Signal generated: {signal}")
            place_trade(signal)
        else:
            print("Position already open. Waiting...")
        time.sleep(60)
except Exception as e:
    print("Error:", e)
finally:
    shutdown()

