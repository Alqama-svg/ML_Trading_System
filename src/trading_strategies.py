"""
Advanced Trading Strategies
Days 3-4: Advanced Trading Strategies
• Statistical arbitrage
• Market-making algorithms
• Cross-asset strategies
• Alternative data integration
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedStrategies:
    def __init__(self):
        self.strategies = {}
        self.positions = {}
        self.performance_metrics = {}
        
    def statistical_arbitrage(self, data_dict, lookback_window=60, entry_threshold=2.0, exit_threshold=0.5):
        """
        Statistical Arbitrage Strategy - Pairs Trading
        
        Args:
            data_dict: Dictionary of DataFrames with price data for different assets
            lookback_window: Window for calculating spread statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
        """
        if len(data_dict) < 2:
            raise ValueError("Need at least 2 assets for pairs trading")
        
        results = {}
        asset_pairs = list(data_dict.keys())
        
        # Find the best correlated pairs
        for i in range(len(asset_pairs)):
            for j in range(i+1, len(asset_pairs)):
                asset1, asset2 = asset_pairs[i], asset_pairs[j]
                
                # Get common dates
                df1 = data_dict[asset1]
                df2 = data_dict[asset2]
                
                common_dates = df1.index.intersection(df2.index)
                if len(common_dates) < lookback_window:
                    continue
                
                prices1 = df1.loc[common_dates, 'Close']
                prices2 = df2.loc[common_dates, 'Close']
                
                # Calculate correlation
                correlation = prices1.corr(prices2)
                
                if correlation > 0.7:  # High correlation threshold
                    signals = self._pairs_trading_signals(
                        prices1, prices2, lookback_window, entry_threshold, exit_threshold
                    )
                    
                    results[f"{asset1}_{asset2}"] = {
                        'correlation': correlation,
                        'signals': signals,
                        'spread_stats': self._calculate_spread_stats(prices1, prices2, lookback_window)
                    }
        
        return results
    
    def _pairs_trading_signals(self, prices1, prices2, lookback_window, entry_threshold, exit_threshold):
        """Generate pairs trading signals"""
        # Calculate hedge ratio using linear regression
        hedge_ratios = []
        spreads = []
        z_scores = []
        signals = []
        
        for i in range(lookback_window, len(prices1)):
            # Get historical data for regression
            y = prices1.iloc[i-lookback_window:i].values
            x = prices2.iloc[i-lookback_window:i].values
            
            # Calculate hedge ratio (beta)
            hedge_ratio = np.cov(y, x)[0, 1] / np.var(x)
            hedge_ratios.append(hedge_ratio)
            
            # Calculate spread
            spread = prices1.iloc[i] - hedge_ratio * prices2.iloc[i]
            spreads.append(spread)
            
            # Calculate z-score of spread
            if len(spreads) >= lookbook_window:
                spread_mean = np.mean(spreads[-lookback_window:])
                spread_std = np.std(spreads[-lookback_window:])
                z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0
            else:
                z_score = 0
            
            z_scores.append(z_score)
            
            # Generate signals
            if abs(z_score) > entry_threshold:
                if z_score > 0:
                    signal = 'SHORT_SPREAD'  # Short asset1, Long asset2
                else:
                    signal = 'LONG_SPREAD'   # Long asset1, Short asset2
            elif abs(z_score) < exit_threshold:
                signal = 'CLOSE_POSITION'
            else:
                signal = 'HOLD'
            
            signals.append(signal)
        
        return pd.DataFrame({
            'hedge_ratio': hedge_ratios,
            'spread': spreads,
            'z_score': z_scores,
            'signal': signals
        }, index=prices1.index[lookback_window:])
    
    def _calculate_spread_stats(self, prices1, prices2, window):
        """Calculate spread statistics"""
        hedge_ratio = np.cov(prices1, prices2)[0, 1] / np.var(prices2)
        spread = prices1 - hedge_ratio * prices2
        
        return {
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'half_life': self._calculate_half_life(spread),
            'adf_pvalue': self._adf_test(spread)
        }
    
    def _calculate_half_life(self, spread, max_lag=50):
        """Calculate half-life of mean reversion"""
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        # Align series
        common_idx = spread_diff.index.intersection(spread_lag.index)
        spread_diff = spread_diff.loc[common_idx]
        spread_lag = spread_lag.loc[common_idx]
        
        if len(spread_diff) < 10:
            return np.nan
        
        # Run regression: Δspread = α + β * spread_t-1
        slope, intercept, r_value, p_value, std_err = stats.linregress(spread_lag, spread_diff)
        
        if slope >= 0:
            return np.nan  # No mean reversion
        
        half_life = -np.log(2) / slope
        return half_life
    
    def _adf_test(self, series):
        """Augmented Dickey-Fuller test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna())
            return result[1]  # p-value
        except:
            return np.nan
    
    def market_making_strategy(self, data, spread_threshold=0.001, inventory_limit=1000):
        """
        Market Making Strategy
        
        Args:
            data: Price data DataFrame
            spread_threshold: Minimum spread to place orders
            inventory_limit: Maximum inventory position
        """
        signals = []
        inventory = 0
        cash = 0
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            prev_price = data['Close'].iloc[i-1]
            volume = data['Volume'].iloc[i]
            
            # Calculate bid-ask spread (simplified using high-low)
            spread = (data['High'].iloc[i] - data['Low'].iloc[i]) / current_price
            
            # Market making logic
            if spread > spread_threshold:
                # Place bid and ask orders
                bid_price = current_price * (1 - spread/2)
                ask_price = current_price * (1 + spread/2)
                
                # Inventory management
                if inventory < inventory_limit and inventory > -inventory_limit:
                    if current_price < prev_price and inventory < inventory_limit:
                        # Buy signal (provide liquidity)
                        signal = 'BUY_LIMIT'
                        inventory += 100  # Assume 100 shares
                        cash -= bid_price * 100
                    elif current_price > prev_price and inventory > -inventory_limit:
                        # Sell signal (provide liquidity)
                        signal = 'SELL_LIMIT'
                        inventory -= 100
                        cash += ask_price * 100
                    else:
                        signal = 'HOLD'
                else:
                    # Inventory too high, reduce position
                    if inventory > 0:
                        signal = 'SELL'
                    elif inventory < 0:
                        signal = 'BUY'
                    else:
                        signal = 'HOLD'
            else:
                signal = 'HOLD'
            
            signals.append({
                'signal': signal,
                'price': current_price,
                'spread': spread,
                'inventory': inventory,
                'cash': cash,
                'pnl': cash + inventory * current_price
            })
        
        return pd.DataFrame(signals, index=data.index[1:])
    
    def cross_asset_momentum(self, data_dict, lookback_period=20, momentum_threshold=0.02):
        """
        Cross-Asset Momentum Strategy
        
        Args:
            data_dict: Dictionary of DataFrames with different asset classes
            lookback_period: Period for momentum calculation
            momentum_threshold: Threshold for momentum signal
        """
        # Calculate momentum for each asset
        asset_momentum = {}
        
        for asset, data in data_dict.items():
            returns = data['Close'].pct_change()
            momentum = returns.rolling(window=lookback_period).mean()
            volatility = returns.rolling(window=lookback_period).std()
            
            # Risk-adjusted momentum
            risk_adj_momentum = momentum / volatility
            
            asset_momentum[asset] = {
                'momentum': momentum,
                'volatility': volatility,
                'risk_adj_momentum': risk_adj_momentum,
                'signals': self._generate_momentum_signals(risk_adj_momentum, momentum_threshold)
            }
        
        # Cross-asset relative momentum
        relative_momentum = self._calculate_relative_momentum(asset_momentum)
        
        return {
            'individual_momentum': asset_momentum,
            'relative_momentum': relative_momentum,
            'portfolio_signals': self._generate_portfolio_signals(relative_momentum)
        }
    
    def _generate_momentum_signals(self, momentum, threshold):
        """Generate momentum-based trading signals"""
        signals = []
        
        for value in momentum:
            if pd.isna(value):
                signals.append('HOLD')
            elif value > threshold:
                signals.append('BUY')
            elif value < -threshold:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return signals
    
    def _calculate_relative_momentum(self, asset_momentum):
        """Calculate relative momentum across assets"""
        # Get common dates
        common_dates = None
        for asset, data in asset_momentum.items():
            if common_dates is None:
                common_dates = data['risk_adj_momentum'].index
            else:
                common_dates = common_dates.intersection(data['risk_adj_momentum'].index)
        
        # Create momentum matrix
        momentum_matrix = pd.DataFrame()
        for asset, data in asset_momentum.items():
            momentum_matrix[asset] = data['risk_adj_momentum'].loc[common_dates]
        
        # Calculate relative momentum (rank-based)
        relative_momentum = momentum_matrix.rank(axis=1, pct=True)
        
        return relative_momentum
    
    def _generate_portfolio_signals(self, relative_momentum):
        """Generate portfolio allocation signals based on relative momentum"""
        signals = {}
        
        for date in relative_momentum.index:
            ranks = relative_momentum.loc[date]
            
            # Top 20% get positive allocation
            top_threshold = ranks.quantile(0.8)
            bottom_threshold = ranks.quantile(0.2)
            
            date_signals = {}
            for asset, rank in ranks.items():
                if rank >= top_threshold:
                    date_signals[asset] = 'OVERWEIGHT'
                elif rank <= bottom_threshold:
                    date_signals[asset] = 'UNDERWEIGHT'
                else:
                    date_signals[asset] = 'NEUTRAL'
            
            signals[date] = date_signals
        
        return signals
    
    def mean_reversion_strategy(self, data, lookback_window=20, entry_z_score=2.0, exit_z_score=0.5):
        """
        Mean Reversion Strategy
        
        Args:
            data: Price data DataFrame
            lookback_window: Window for calculating mean and std
            entry_z_score: Z-score threshold for entry
            exit_z_score: Z-score threshold for exit
        """
        prices = data['Close']
        signals = []
        position = 0  # 0: no position, 1: long, -1: short
        
        for i in range(lookback_window, len(prices)):
            # Calculate rolling statistics
            window_prices = prices.iloc[i-lookback_window:i]
            mean_price = window_prices.mean()
            std_price = window_prices.std()
            
            current_price = prices.iloc[i]
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            # Generate signals
            if position == 0:  # No current position
                if z_score > entry_z_score:
                    signal = 'SELL'  # Price too high, expect reversion
                    position = -1
                elif z_score < -entry_z_score:
                    signal = 'BUY'   # Price too low, expect reversion
                    position = 1
                else:
                    signal = 'HOLD'
            else:  # Currently in position
                if position == 1 and z_score > -exit_z_score:  # Long position
                    signal = 'SELL'
                    position = 0
                elif position == -1 and z_score < exit_z_score:  # Short position
                    signal = 'BUY'
                    position = 0
                else:
                    signal = 'HOLD'
            
            signals.append({
                'signal': signal,
                'z_score': z_score,
                'position': position,
                'price': current_price,
                'mean_price': mean_price,
                'std_price': std_price
            })
        
        return pd.DataFrame(signals, index=data.index[lookback_window:])
    
    def volatility_targeting_strategy(self, data, target_volatility=0.15, lookback_window=30):
        """
        Volatility Targeting Strategy
        Adjusts position size based on realized volatility
        
        Args:
            data: Price data DataFrame
            target_volatility: Target annualized volatility
            lookback_window: Window for volatility calculation
        """
        returns = data['Close'].pct_change()
        signals = []
        
        for i in range(lookback_window, len(returns)):
            # Calculate realized volatility
            window_returns = returns.iloc[i-lookback_window:i]
            realized_vol = window_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate position sizing multiplier
            if realized_vol > 0:
                vol_multiplier = target_volatility / realized_vol
                vol_multiplier = min(max(vol_multiplier, 0.1), 3.0)  # Cap between 0.1 and 3.0
            else:
                vol_multiplier = 1.0
            
            # Base signal (can be replaced with any strategy)
            base_signal = 'BUY' if returns.iloc[i-1] > 0 else 'SELL'
            
            signals.append({
                'base_signal': base_signal,
                'vol_multiplier': vol_multiplier,
                'realized_vol': realized_vol,
                'target_vol': target_volatility,
                'adjusted_signal': base_signal if vol_multiplier > 0.5 else 'HOLD'
            })
        
        return pd.DataFrame(signals, index=data.index[lookback_window:])

class AlternativeDataIntegration:
    """
    Integration of alternative data sources
    """
    
    def __init__(self):
        self.data_sources = {}
        
    def sentiment_analysis(self, text_data):
        """
        Simple sentiment analysis (placeholder for more sophisticated NLP)
        
        Args:
            text_data: List of text strings (news, tweets, etc.)
        """
        # Simplified sentiment scoring
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'rise', 'gain']
        negative_words = ['bad', 'terrible', 'negative', 'bearish', 'down', 'fall', 'loss', 'drop']
        
        sentiments = []
        
        for text in text_data:
            if isinstance(text, str):
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                # Calculate sentiment score
                total_words = len(text_lower.split())
                if total_words > 0:
                    sentiment_score = (positive_count - negative_count) / total_words
                else:
                    sentiment_score = 0
                
                sentiments.append(sentiment_score)
            else:
                sentiments.append(0)
        
        return sentiments
    
    def social_media_signals(self, social_data):
        """
        Extract trading signals from social media data
        
        Args:
            social_data: DataFrame with social media metrics (mentions, likes, shares, etc.)
        """
        signals = []
        
        for i in range(len(social_data)):
            mentions = social_data.get('mentions', [0])[i] if i < len(social_data.get('mentions', [])) else 0
            sentiment = social_data.get('sentiment', [0])[i] if i < len(social_data.get('sentiment', [])) else 0
            
            # Normalize metrics
            mention_score = min(mentions / 1000, 1.0)  # Cap at 1000 mentions
            sentiment_score = max(-1, min(1, sentiment))  # Cap between -1 and 1
            
            # Combine signals
            combined_signal = (mention_score * 0.3) + (sentiment_score * 0.7)
            
            if combined_signal > 0.3:
                signal = 'BUY'
            elif combined_signal < -0.3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append({
                'signal': signal,
                'mention_score': mention_score,
                'sentiment_score': sentiment_score,
                'combined_signal': combined_signal
            })
        
        return signals
    
    def economic_indicators_signals(self, economic_data):
        """
        Generate signals based on economic indicators
        
        Args:
            economic_data: DataFrame with economic indicators (GDP, inflation, unemployment, etc.)
        """
        signals = []
        
        # Define indicator weights and thresholds
        indicator_weights = {
            'gdp_growth': 0.3,
            'inflation_rate': -0.2,  # Negative weight (high inflation = negative)
            'unemployment_rate': -0.25,  # Negative weight
            'interest_rate': -0.15,  # Negative weight for stocks
            'consumer_confidence': 0.1
        }
        
        for i in range(len(economic_data)):
            total_signal = 0
            
            for indicator, weight in indicator_weights.items():
                if indicator in economic_data.columns:
                    value = economic_data[indicator].iloc[i]
                    
                    # Normalize indicators (simplified)
                    if indicator == 'gdp_growth':
                        normalized_value = min(max(value / 5.0, -1), 1)  # Normalize around 5% growth
                    elif indicator == 'inflation_rate':
                        normalized_value = min(max((2.0 - value) / 2.0, -1), 1)  # Target 2% inflation
                    elif indicator == 'unemployment_rate':
                        normalized_value = min(max((5.0 - value) / 5.0, -1), 1)  # Target 5% unemployment
                    elif indicator == 'interest_rate':
                        normalized_value = min(max((3.0 - value) / 3.0, -1), 1)  # Normalize around 3%
                    elif indicator == 'consumer_confidence':
                        normalized_value = min(max((value - 50) / 50, -1), 1)  # Index around 100
                    else:
                        normalized_value = 0
                    
                    total_signal += normalized_value * weight
            
            # Generate signal
            if total_signal > 0.2:
                signal = 'BUY'
            elif total_signal < -0.2:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append({
                'signal': signal,
                'economic_score': total_signal
            })
        
        return signals

class PortfolioOptimizer:
    """
    Portfolio optimization using various methods
    """
    
    def __init__(self):
        self.weights = {}
        self.expected_returns = {}
        self.covariance_matrix = None
        
    def mean_variance_optimization(self, returns_data, risk_aversion=1.0):
        """
        Mean-variance optimization (Markowitz)
        
        Args:
            returns_data: DataFrame with asset returns
            risk_aversion: Risk aversion parameter
        """
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean()
        cov_matrix = returns_data.cov()
        
        n_assets = len(expected_returns)
        
        # Objective function: minimize -expected_return + risk_aversion * variance
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -portfolio_return + risk_aversion * portfolio_variance
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = dict(zip(returns_data.columns, result.x))
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(result.x, expected_returns)
            portfolio_variance = np.dot(result.x, np.dot(cov_matrix, result.x))
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio
            }
        else:
            return None
    
    def risk_parity_optimization(self, returns_data):
        """
        Risk parity optimization - equal risk contribution
        
        Args:
            returns_data: DataFrame with asset returns
        """
        cov_matrix = returns_data.cov().values
        n_assets = len(returns_data.columns)
        
        def risk_budget_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Calculate marginal risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Calculate risk contributions
            risk_contrib = weights * marginal_contrib
            
            # Target equal risk contribution
            target_risk = portfolio_vol / n_assets
            
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0.01, 0.5) for _ in range(n_assets)]  # Min 1%, max 50%
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(returns_data.columns, result.x))
            return weights
        else:
            return None
    
    def black_litterman_optimization(self, returns_data, market_caps, views=None, confidence=None):
        """
        Black-Litterman optimization
        
        Args:
            returns_data: DataFrame with asset returns
            market_caps: Dictionary with market capitalizations
            views: Dictionary with investor views on expected returns
            confidence: Confidence level in views
        """
        # Market equilibrium weights (based on market caps)
        total_market_cap = sum(market_caps.values())
        market_weights = np.array([market_caps.get(asset, 0) / total_market_cap 
                                 for asset in returns_data.columns])
        
        # Historical covariance matrix
        cov_matrix = returns_data.cov().values
        
        # Risk aversion parameter (estimated)
        risk_aversion = 3.0
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views is None or confidence is None:
            # No views, return market weights
            return dict(zip(returns_data.columns, market_weights))
        
        # Views matrix P and view vector Q
        n_assets = len(returns_data.columns)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        
        for i, (asset, view_return) in enumerate(views.items()):
            asset_idx = list(returns_data.columns).index(asset)
            P[i, asset_idx] = 1
            Q[i] = view_return
        
        # Uncertainty matrix for views
        tau = 0.025  # Scaling factor
        omega = confidence * np.dot(P, np.dot(tau * cov_matrix, P.T))
        
        # Black-Litterman formula
        M1 = np.linalg.inv(tau * cov_matrix)
        M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
        M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
        M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
        
        # New expected returns
        mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
        
        # New covariance matrix
        cov_bl = np.linalg.inv(M1 + M2)
        
        # Optimize portfolio
        def objective(weights):
            portfolio_return = np.dot(weights, mu_bl)
            portfolio_variance = np.dot(weights, np.dot(cov_bl, weights))
            return -portfolio_return + risk_aversion * portfolio_variance
        
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(0, 1) for _ in range(n_assets)]
        x0 = market_weights
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return dict(zip(returns_data.columns, result.x))
        else:
            return dict(zip(returns_data.columns, market_weights))

class RiskManagement:
    """
    Advanced risk management techniques
    """
    
    def __init__(self):
        self.risk_metrics = {}
        
    def calculate_var(self, returns, confidence_level=0.05):
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (default 5%)
        """
        if len(returns) == 0:
            return 0
        
        # Historical VaR
        historical_var = np.percentile(returns, confidence_level * 100)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        parametric_var = mean_return - stats.norm.ppf(1 - confidence_level) * std_return
        
        # Modified VaR (Cornish-Fisher expansion)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        z_score = stats.norm.ppf(1 - confidence_level)
        z_cf = (z_score + 
                (z_score**2 - 1) * skewness / 6 + 
                (z_score**3 - 3*z_score) * kurtosis / 24 - 
                (2*z_score**3 - 5*z_score) * skewness**2 / 36)
        
        modified_var = mean_return - z_cf * std_return
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'modified_var': modified_var
        }
    
    def calculate_expected_shortfall(self, returns, confidence_level=0.05):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level (default 5%)
        """
        var_threshold = np.percentile(returns, confidence_level * 100)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) > 0:
            expected_shortfall = tail_returns.mean()
        else:
            expected_shortfall = var_threshold
        
        return expected_shortfall
    
    def calculate_maximum_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Series of portfolio values
        """
        if len(equity_curve) == 0:
            return 0, 0, 0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        
        # Find peak before max drawdown
        peak_idx = running_max.loc[:max_drawdown_idx].idxmax()
        
<<<<<<< HEAD
        return max_drawdown, peak_idx, max_drawdown_idx
=======
        return max_drawdown, peak_idx, max_drawdown_idx
>>>>>>> f1909685739746bbe77927120694a7980b73754a
