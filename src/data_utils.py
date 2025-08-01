import pandas as pd
import yfinance as yf
from collections import deque
import numpy as np

# Global buffer for real-time feature calculation
DATA_BUFFER_SIZE = 100 # Adjust based on your longest rolling window + lag requirements
data_buffer = deque(maxlen=DATA_BUFFER_SIZE)

def mock_realtime_data_feed(ticker="NVDA", period="7d", interval="1m"):
    
    print(f"Fetching historical data for {ticker} to simulate real-time feed...")
    df_hist = yf.Ticker(ticker).history(period=period, interval=interval)
    df_hist.dropna(inplace=True)
    print(f"Loaded {len(df_hist)} historical data points for simulation.")

    for index, row in df_hist.iterrows():
        yield index, row # Yield timestamp and data row

def initialize_data_buffer(ticker="NVDA", period="7d", interval="1m"):
    
    print(f"Filling initial data buffer ({DATA_BUFFER_SIZE} points required)...")
    initial_df_for_buffer = yf.Ticker(ticker).history(period=period, interval=interval)
    initial_df_for_buffer.dropna(inplace=True)

    initial_fill_count = min(len(initial_df_for_buffer), DATA_BUFFER_SIZE)

    for i in range(initial_fill_count):
        data_buffer.append(initial_df_for_buffer.iloc[i])
    print(f"Buffer filled with {len(data_buffer)} initial data points.")

    data_generator = mock_realtime_data_feed(ticker=ticker, period=period, interval=interval)
    for _ in range(initial_fill_count):
        try:
            next(data_generator)
        except StopIteration:
            print("Warning: Data generator exhausted while skipping initial buffer points.")
            break
    return data_generator

# Define the features that our model expects (must match training)
FEATURES = [
    'macd', 'macd_signal', 'rolling_std_20', 'rsi', 'signed_volume',
    'log_return_lag_1', 'rsi_lag_1', 'log_return_lag_2', 'rsi_lag_2',
    'log_return_lag_3', 'rsi_lag_3', 'log_return_lag_5', 'rsi_lag_5',
    'log_return_lag_10', 'rsi_lag_10', 'hour', 'minute'
]