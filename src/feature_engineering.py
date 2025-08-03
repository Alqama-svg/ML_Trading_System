"""
Feature Engineering for Financial Data
Advanced technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_features(data, include_advanced=True):
    """
    Create comprehensive features for ML trading models
    
    Args:
        data: DataFrame with OHLCV data
        include_advanced: Whether to include advanced features
    
    Returns:
        DataFrame with engineered features
    """
    if data is None or data.empty:
        return pd.DataFrame()
    
    df = data.copy()
    
    # Basic price features
    df = add_basic_features(df)
    
    # Technical indicators
    df = add_technical_indicators(df)
    
    # Volume features
    df = add_volume_features(df)
    
    # Volatility features
    df = add_volatility_features(df)
    
    if include_advanced:
        # Advanced features
        df = add_advanced_features(df)
        
        # Market microstructure features
        df = add_microstructure_features(df)
        
        # Time-based features
        df = add_time_features(df)
    
    # Remove infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    return df

def add_basic_features(df):
    """Add basic price-based features"""
    
    # Price changes
    df['price_change'] = df['Close'].pct_change()
    df['price_change_1h'] = df['Close'].pct_change(periods=1)
    df['price_change_4h'] = df['Close'].pct_change(periods=4)
    df['price_change_1d'] = df['Close'].pct_change(periods=24) if len(df) > 24 else df['Close'].pct_change()
    
    # High-Low ratios
    df['hl_ratio'] = df['High'] / df['Low']
    df['oc_ratio'] = df['Open'] / df['Close']
    
    # Price position within range
    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Gap analysis
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    return df

def add_technical_indicators(df):
    """Add technical indicators"""
    
    # Moving averages
    for period in [5, 10, 20, 50]:
        df[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        
        # Price relative to MA
        df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
    
    # MA crossovers
    df['sma_10_20_cross'] = np.where(df['sma_10'] > df['sma_20'], 1, 0)
    df['ema_10_20_cross'] = np.where(df['ema_10'] > df['ema_20'], 1, 0)
    
    # RSI
    df['rsi'] = calculate_rsi(df['Close'])
    df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
    df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic Oscillator
    df['stoch_k'] = calculate_stochastic(df)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = calculate_williams_r(df)
    
    # Average True Range (ATR)
    df['atr'] = calculate_atr(df)
    
    return df

def add_volume_features(df):
    """Add volume-based features"""
    
    # Volume moving averages
    df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
    df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
    
    # Volume ratios
    df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
    df['volume_price_trend'] = df['Volume'] * df['price_change']
    
    # On Balance Volume (OBV)
    df['obv'] = calculate_obv(df)
    
    # Volume Rate of Change
    df['volume_roc'] = df['Volume'].pct_change(periods=10)
    
    # Accumulation/Distribution Line
    df['ad_line'] = calculate_ad_line(df)
    
    # Money Flow Index
    df['mfi'] = calculate_mfi(df)
    
    return df

def add_volatility_features(df):
    """Add volatility-based features"""
    
    # Rolling volatility
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['price_change'].rolling(window=period).std()
        df[f'volatility_{period}_annualized'] = df[f'volatility_{period}'] * np.sqrt(252)
    
    # Volatility ratios
    df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
    
    # Parkinson volatility (using High-Low)
    df['parkinson_vol'] = np.sqrt(
        (1/(4*np.log(2))) * np.log(df['High']/df['Low'])**2
    ).rolling(window=20).mean()
    
    # Garman-Klass volatility
    df['gk_vol'] = calculate_garman_klass_volatility(df)
    
    # True Range
    df['true_range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    
    return df

def add_advanced_features(df):
    """Add advanced statistical features"""
    
    # Momentum indicators
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        df[f'roc_{period}'] = df['Close'].pct_change(periods=period)
    
    # Price acceleration
    df['price_acceleration'] = df['price_change'].diff()
    
    # Fractal dimension
    df['fractal_dimension'] = calculate_fractal_dimension(df['Close'])
    
    # Hurst exponent (simplified)
    df['hurst_exponent'] = calculate_hurst_exponent(df['Close'])
    
    # Z-score of price
    for period in [20, 50]:
        rolling_mean = df['Close'].rolling(window=period).mean()
        rolling_std = df['Close'].rolling(window=period).std()
        df[f'price_zscore_{period}'] = (df['Close'] - rolling_mean) / rolling_std
    
    # Skewness and Kurtosis
    for period in [20, 50]:
        df[f'returns_skew_{period}'] = df['price_change'].rolling(window=period).skew()
        df[f'returns_kurtosis_{period}'] = df['price_change'].rolling(window=period).kurt()
    
    # Correlation with lagged prices
    for lag in [1, 5, 10]:
        df[f'autocorr_lag_{lag}'] = df['price_change'].rolling(window=50).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
        )
    
    return df

def add_microstructure_features(df):
    """Add market microstructure features"""
    
    # Bid-Ask spread proxy (High-Low as proxy)
    df['spread_proxy'] = (df['High'] - df['Low']) / df['Close']
    
    # Tick direction (simplified)
    df['tick_direction'] = np.where(df['Close'] > df['Close'].shift(1), 1, 
                                   np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    
    # Price impact
    df['price_impact'] = abs(df['price_change']) / (df['Volume'] + 1e-10)
    
    # Amihud illiquidity ratio
    df['amihud_illiquidity'] = abs(df['price_change']) / (df['Volume'] * df['Close'] + 1e-10)
    
    # Roll spread estimator
    df['roll_spread'] = 2 * np.sqrt(abs(df['price_change'].rolling(window=20).cov(df['price_change'].shift(1))))
    
    return df

def add_time_features(df):
    """Add time-based features"""
    
    if hasattr(df.index, 'hour'):
        # Hour of day
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Month
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# Helper functions for technical indicators

def calculate_rsi(prices, window=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stochastic(df, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    k_percent = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
    return k_percent

def calculate_williams_r(df, period=14):
    """Calculate Williams %R"""
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    wr = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return wr

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    return true_range.rolling(window=period).mean()

def calculate_obv(df):
    """Calculate On Balance Volume"""
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=df.index)

def calculate_ad_line(df):
    """Calculate Accumulation/Distribution Line"""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    clv = clv.fillna(0)  # Handle division by zero
    ad_line = (clv * df['Volume']).cumsum()
    return ad_line

def calculate_mfi(df, period=14):
    """Calculate Money Flow Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def calculate_garman_klass_volatility(df, period=20):
    """Calculate Garman-Klass volatility estimator"""
    log_hl = np.log(df['High'] / df['Low'])
    log_cc = np.log(df['Close'] / df['Close'].shift(1))
    
    gk_vol = np.sqrt(
        0.5 * (log_hl**2).rolling(window=period).mean() - 
        (2*np.log(2) - 1) * (log_cc**2).rolling(window=period).mean()
    )
    
    return gk_vol

def calculate_fractal_dimension(prices, window=20):
    """Calculate fractal dimension (simplified Higuchi method)"""
    def higuchi_fd(x, k_max=10):
        if len(x) < k_max:
            return np.nan
        
        n = len(x)
        lk = np.zeros(k_max)
        
        for k in range(1, k_max + 1):
            lm = []
            for m in range(k):
                ll = 0
                for i in range(1, int((n - m - 1) / k) + 1):
                    ll += abs(x[m + i * k] - x[m + (i - 1) * k])
                ll = ll * (n - 1) / (((n - m - 1) // k) * k) / k
                lm.append(ll)
            lk[k - 1] = np.mean(lm)
        
        # Fit line to log-log plot
        x_vals = np.log(range(1, k_max + 1))
        y_vals = np.log(lk)
        
        if np.any(np.isinf(y_vals)) or np.any(np.isnan(y_vals)):
            return np.nan
        
        slope, _ = np.polyfit(x_vals, y_vals, 1)
        return -slope
    
    return prices.rolling(window=window).apply(lambda x: higuchi_fd(x.values))

def calculate_hurst_exponent(prices, window=50):
    """Calculate Hurst exponent (simplified R/S method)"""
    def hurst_rs(x):
        if len(x) < 10:
            return np.nan
        
        n = len(x)
        y = np.cumsum(x - np.mean(x))
        r = np.max(y) - np.min(y)
        s = np.std(x)
        
        if s == 0:
            return np.nan
        
        rs = r / s
        return np.log(rs) / np.log(n)
    
    return prices.pct_change().rolling(window=window).apply(lambda x: hurst_rs(x.values))

class FeatureSelector:
    """Feature selection utilities"""
    
    def __init__(self):
        self.selected_features = []
        self.feature_importance = {}
    
    def select_features_by_importance(self, X, y, model, top_k=50):
        """Select top-k features by importance"""
        model.fit(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model doesn't have feature importance or coefficients")
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top-k
        self.selected_features = [f[0] for f in sorted_features[:top_k]]
        self.feature_importance = dict(sorted_features[:top_k])
        
        return self.selected_features
    
    def select_features_by_correlation(self, X, threshold=0.95):
        """Remove highly correlated features"""
        correlation_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for pair in high_corr_pairs:
            features_to_remove.add(pair[1])  # Remove the second feature
        
        selected_features = [col for col in X.columns if col not in features_to_remove]
        
        return selected_features