import pandas as pd
import numpy as np
from src.data_utils import FEATURES # Import FEATURES from data_utils

def compute_realtime_features(new_data_point_row, data_buffer_deque):
    """
    Computes features for a new incoming data point using a rolling data buffer.
    """
    # Append the new data point to the buffer
    data_buffer_deque.append(new_data_point_row)

    # Convert deque to DataFrame for easy Pandas calculations
    temp_df = pd.DataFrame(list(data_buffer_deque))
    temp_df.index = pd.to_datetime(temp_df.index) # Ensure index is DatetimeIndex

    # Ensure essential columns exist (Open, High, Low, Close, Volume)
    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_ohlcv:
        if col not in temp_df.columns:
            temp_df[col] = np.nan # Placeholder if not present

    # --- Feature Engineering ---
    # Log Returns
    temp_df['log_return'] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))

    # Momentum Features (MACD)
    temp_df['ema_12'] = temp_df['Close'].ewm(span=12, adjust=False).mean()
    temp_df['ema_26'] = temp_df['Close'].ewm(span=26, adjust=False).mean()
    temp_df['macd'] = temp_df['ema_12'] - temp_df['ema_26']
    temp_df['macd_signal'] = temp_df['macd'].ewm(span=9, adjust=False).mean()

    # Volatility features (Bollinger Bands components)
    temp_df['rolling_std_20'] = temp_df['Close'].rolling(window=20).std()
    temp_df['bollinger_high'] = temp_df['Close'].rolling(window=20).mean() + (temp_df['rolling_std_20'] * 2)
    temp_df['bollinger_low'] = temp_df['Close'].rolling(window=20).mean() - (temp_df['rolling_std_20'] * 2)

    # Oscillator features (RSI)
    delta = temp_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain / loss
        temp_df['rsi'] = 100 - (100 / (1 + rs))
    temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Order flow proxy
    temp_df['signed_volume'] = temp_df['Volume'] * np.sign(temp_df['log_return'])

    # Lagged features
    for i in [1, 2, 3, 5, 10]:
        temp_df[f'log_return_lag_{i}'] = temp_df['log_return'].shift(i)
        temp_df[f'rsi_lag_{i}'] = temp_df['rsi'].shift(i)

    # Time-based features
    if not isinstance(temp_df.index, pd.DatetimeIndex):
        temp_df.index = pd.to_datetime(temp_df.index)
    temp_df['hour'] = temp_df.index.hour
    temp_df['minute'] = temp_df.index.minute

    # Get the latest computed features. Use .tail(1) to get the last row as a DataFrame
    latest_features_df = temp_df[FEATURES].tail(1)

    if latest_features_df.isnull().values.any():
        return None

    return latest_features_df