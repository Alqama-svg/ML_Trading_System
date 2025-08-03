"""
Risk Management System
Advanced risk controls and portfolio protection
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.risk_limits = self._set_default_risk_limits()
        self.alerts = []
        
    def _set_default_risk_limits(self):
        """Set default risk limits"""
        return {
            'max_position_size': 0.1,  # 10% max position size
            'max_portfolio_risk': 0.02,  # 2% daily portfolio risk
            'max_drawdown': 0.15,  # 15% max drawdown
            'max_leverage': 2.0,  # 2x max leverage
            'max_correlation_exposure': 0.3,  # 30% max in correlated assets
            'var_limit': 0.05,  # 5% VaR limit
            'stop_loss_pct': 0.05,  # 5% stop loss
            'position_timeout': 30  # Max 30 days per position
        }
    
    def update_risk_limits(self, new_limits):
        """Update risk limits"""
        self.risk_limits.update(new_limits)
        
    def check_position_size(self, symbol, proposed_size, current_price):
        """
        Check if proposed position size is within risk limits
        
        Args:
            symbol: Trading symbol
            proposed_size: Proposed position size (number of shares)
            current_price: Current market price
        
        Returns:
            (bool, str): (is_valid, message)
        """
        position_value = abs(proposed_size * current_price)
        position_pct = position_value / self.current_capital
        
        if position_pct > self.risk_limits['max_position_size']:
            return False, f"Position size {position_pct:.2%} exceeds limit {self.risk_limits['max_position_size']:.2%}"
        
        return True, "Position size acceptable"
    
    def check_portfolio_risk(self, positions, prices, returns_data=None):
        """
        Check overall portfolio risk
        
        Args:
            positions: Dictionary of {symbol: shares}
            prices: Dictionary of {symbol: current_price}
            returns_data: DataFrame of historical returns
        
        Returns:
            dict: Risk metrics and alerts
        """
        risk_metrics = {}
        alerts = []
        
        # Calculate portfolio value
        portfolio_value = sum(positions.get(symbol, 0) * prices.get(symbol, 0) 
                            for symbol in set(positions.keys()) | set(prices.keys()))
        
        # Calculate leverage
        total_exposure = sum(abs(positions.get(symbol, 0) * prices.get(symbol, 0)) 
                           for symbol in set(positions.keys()) | set(prices.keys()))
        leverage = total_exposure / self.current_capital if self.current_capital > 0 else 0
        
        risk_metrics['portfolio_value'] = portfolio_value
        risk_metrics['leverage'] = leverage
        risk_metrics['cash'] = self.current_capital - portfolio_value
        
        # Check leverage limit
        if leverage > self.risk_limits['max_leverage']:
            alerts.append(f"Leverage {leverage:.2f}x exceeds limit {self.risk_limits['max_leverage']:.2f}x")
        
        # Calculate VaR if returns data available
        if returns_data is not None and not returns_data.empty:
            portfolio_var = self.calculate_portfolio_var(positions, returns_data)
            risk_metrics['var_95'] = portfolio_var
            
            if abs(portfolio_var) > self.risk_limits['var_limit']:
                alerts.append(f"Portfolio VaR {portfolio_var:.2%} exceeds limit {self.risk_limits['var_limit']:.2%}")
        
        # Calculate concentration risk
        concentration_risk = self.calculate_concentration_risk(positions, prices)
        risk_metrics['concentration_risk'] = concentration_risk
        
        # Calculate correlation risk if data available
        if returns_data is not None:
            correlation_risk = self.calculate_correlation_risk(positions, returns_data)
            risk_metrics['correlation_risk'] = correlation_risk
        
        self.alerts.extend(alerts)
        return risk_metrics
    
    def calculate_portfolio_var(self, positions, returns_data, confidence_level=0.05):
        """
        Calculate portfolio Value at Risk
        
        Args:
            positions: Dictionary of positions
            returns_data: DataFrame of historical returns
            confidence_level: VaR confidence level
        
        Returns:
            float: Portfolio VaR
        """
        if returns_data.empty or not positions:
            return 0
        
        # Get common assets
        common_assets = set(positions.keys()) & set(returns_data.columns)
        if not common_assets:
            return 0
        
        # Calculate portfolio weights
        total_value = sum(abs(pos) for pos in positions.values())
        if total_value == 0:
            return 0
        
        weights = np.array([positions.get(asset, 0) / total_value for asset in common_assets])
        returns_matrix = returns_data[list(common_assets)].values
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Calculate VaR
        var_95 = np.percentile(portfolio_returns, confidence_level * 100)
        
        return var_95
    
    def calculate_concentration_risk(self, positions, prices):
        """
        Calculate concentration risk (Herfindahl-Hirschman Index)
        
        Args:
            positions: Dictionary of positions
            prices: Dictionary of prices
        
        Returns:
            float: Concentration index (0-1, higher = more concentrated)
        """
        if not positions or not prices:
            return 0
        
        # Calculate position values
        position_values = {}
        total_value = 0
        
        for symbol, shares in positions.items():
            if symbol in prices:
                value = abs(shares * prices[symbol])
                position_values[symbol] = value
                total_value += value
        
        if total_value == 0:
            return 0
        
        # Calculate concentration index
        weights_squared = sum((value / total_value) ** 2 for value in position_values.values())
        
        return weights_squared
    
    def calculate_correlation_risk(self, positions, returns_data):
        """
        Calculate correlation risk in portfolio
        
        Args:
            positions: Dictionary of positions
            returns_data: DataFrame of historical returns
        
        Returns:
            dict: Correlation risk metrics
        """
        common_assets = set(positions.keys()) & set(returns_data.columns)
        
        if len(common_assets) < 2:
            return {'avg_correlation': 0, 'max_correlation': 0, 'correlation_clusters': []}
        
        # Calculate correlation matrix
        corr_matrix = returns_data[list(common_assets)].corr()
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlations = upper_triangle.stack().values
        
        # Calculate metrics
        avg_correlation = np.mean(correlations)
        max_correlation = np.max(correlations)
        
        # Find highly correlated clusters
        high_corr_threshold = 0.7
        correlation_clusters = []
        
        for i, asset1 in enumerate(common_assets):
            for j, asset2 in enumerate(common_assets):
                if i < j and abs(corr_matrix.loc[asset1, asset2]) > high_corr_threshold:
                    correlation_clusters.append((asset1, asset2, corr_matrix.loc[asset1, asset2]))
        
        return {
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'correlation_clusters': correlation_clusters
        }
    
    def apply_stop_loss(self, positions, current_prices, entry_prices):
        """
        Apply stop-loss rules
        
        Args:
            positions: Current positions
            current_prices: Current market prices
            entry_prices: Entry prices for positions
        
        Returns:
            list: Stop-loss signals
        """
        stop_loss_signals = []
        
        for symbol in positions:
            if symbol not in current_prices or symbol not in entry_prices:
                continue
            
            position_size = positions[symbol]
            current_price = current_prices[symbol]
            entry_price = entry_prices[symbol]
            
            if position_size == 0:
                continue
            
            # Calculate return
            if position_size > 0:  # Long position
                return_pct = (current_price - entry_price) / entry_price
            else:  # Short position
                return_pct = (entry_price - current_price) / entry_price
            
            # Check stop-loss
            if return_pct < -self.risk_limits['stop_loss_pct']:
                stop_loss_signals.append({
                    'symbol': symbol,
                    'action': 'CLOSE_POSITION',
                    'reason': 'STOP_LOSS',
                    'current_return': return_pct,
                    'stop_loss_limit': -self.risk_limits['stop_loss_pct']
                })
        
        return stop_loss_signals
    
    def apply_position_sizing(self, signal, symbol, current_price, volatility, portfolio_value):
        """
        Calculate optimal position size based on risk
        
        Args:
            signal: Trading signal strength
            symbol: Trading symbol
            current_price: Current market price
            volatility: Asset volatility
            portfolio_value: Current portfolio value
        
        Returns:
            int: Recommended position size (shares)
        """
        # Kelly Criterion-based sizing
        if volatility <= 0:
            return 0
        
        # Estimate win probability and payoff ratio (simplified)
        win_prob = 0.55  # Assume slight edge
        avg_win = 0.02   # 2% average win
        avg_loss = 0.015 # 1.5% average loss
        
        # Kelly fraction
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / (avg_win * avg_loss)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Adjust for volatility
        vol_adjustment = min(1.0, 0.02 / volatility)  # Target 2% volatility
        
        # Calculate position size
        risk_capital = portfolio_value * kelly_fraction * vol_adjustment
        position_size = int(risk_capital / current_price)
        
        # Apply maximum position size limit
        max_shares = int(portfolio_value * self.risk_limits['max_position_size'] / current_price)
        position_size = min(position_size, max_shares)
        
        return position_size
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annualized)
        
        Returns:
            float: Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)
    
    def calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown
        
        Args:
            equity_curve: Series of portfolio values
        
        Returns:
            dict: Drawdown metrics
        """
        if len(equity_curve) == 0:
            return {'max_drawdown': 0, 'current_drawdown': 0, 'drawdown_duration': 0}
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Current drawdown
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown duration (simplified)
        drawdown_periods = (drawdown < -0.01).sum()  # Periods with >1% drawdown
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_date': max_dd_date,
            'current_drawdown': current_drawdown,
            'drawdown_duration': drawdown_periods
        }
    
    def stress_test_portfolio(self, positions, prices, scenarios):
        """
        Perform stress testing on portfolio
        
        Args:
            positions: Current positions
            prices: Current prices
            scenarios: List of stress scenarios
        
        Returns:
            dict: Stress test results
        """
        stress_results = {}
        
        for scenario_name, scenario in scenarios.items():
            portfolio_pnl = 0
            
            for symbol, shares in positions.items():
                if symbol in prices and symbol in scenario:
                    current_price = prices[symbol]
                    stress_price = current_price * (1 + scenario[symbol])
                    position_pnl = shares * (stress_price - current_price)
                    portfolio_pnl += position_pnl
            
            stress_results[scenario_name] = {
                'portfolio_pnl': portfolio_pnl,
                'portfolio_pnl_pct': portfolio_pnl / self.current_capital if self.current_capital > 0 else 0
            }
        
        return stress_results
    
    def generate_risk_report(self, positions, prices, returns_data=None):
        """
        Generate comprehensive risk report
        
        Args:
            positions: Current positions
            prices: Current prices
            returns_data: Historical returns data
        
        Returns:
            dict: Comprehensive risk report
        """
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_summary': {},
            'risk_metrics': {},
            'alerts': self.alerts.copy(),
            'recommendations': []
        }
        
        # Portfolio summary
        portfolio_value = sum(positions.get(symbol, 0) * prices.get(symbol, 0) 
                            for symbol in set(positions.keys()) | set(prices.keys()))
        
        report['portfolio_summary'] = {
            'total_value': portfolio_value,
            'cash': self.current_capital - portfolio_value,
            'number_of_positions': len([p for p in positions.values() if p != 0]),
            'largest_position': max([abs(p * prices.get(s, 0)) for s, p in positions.items()]) if positions else 0
        }
        
        # Risk metrics
        risk_metrics = self.check_portfolio_risk(positions, prices, returns_data)
        report['risk_metrics'] = risk_metrics
        
        # Concentration analysis
        concentration = self.calculate_concentration_risk(positions, prices)
        report['risk_metrics']['concentration_index'] = concentration
        
        # Generate recommendations
        recommendations = []
        
        if concentration > 0.5:
            recommendations.append("High concentration risk - consider diversification")
        
        if risk_metrics.get('leverage', 0) > 1.5:
            recommendations.append("High leverage - consider reducing position sizes")
        
        if len(self.alerts) > 0:
            recommendations.append("Active risk alerts - review position limits")
        
        report['recommendations'] = recommendations
        
        # Clear alerts after report
        self.alerts = []
        
        return report

class DynamicRiskAdjustment:
    """
    Dynamic risk adjustment based on market conditions
    """
    
    def __init__(self, base_risk_manager):
        self.base_manager = base_risk_manager
        self.market_regime = 'normal'
        self.volatility_adjustment = 1.0
        
    def adjust_risk_limits_for_regime(self, market_regime, volatility_level):
        """
        Adjust risk limits based on market regime
        
        Args:
            market_regime: 'bull', 'bear', 'sideways', 'volatile'
            volatility_level: Current volatility level (0-1 scale)
        """
        self.market_regime = market_regime
        self.volatility_adjustment = min(2.0, max(0.5, 1 / (1 + volatility_level)))
        
        # Adjust base risk limits
        adjusted_limits = self.base_manager.risk_limits.copy()
        
        if market_regime == 'volatile':
            # Reduce risk in volatile markets
            adjusted_limits['max_position_size'] *= 0.7
            adjusted_limits['max_portfolio_risk'] *= 0.8
            adjusted_limits['stop_loss_pct'] *= 0.8
            
        elif market_regime == 'bear':
            # Conservative approach in bear markets
            adjusted_limits['max_position_size'] *= 0.6
            adjusted_limits['max_leverage'] *= 0.8
            adjusted_limits['stop_loss_pct'] *= 0.7
            
        elif market_regime == 'bull':
            # Slightly more aggressive in bull markets
            adjusted_limits['max_position_size'] *= 1.2
            adjusted_limits['max_leverage'] *= 1.1
            
        # Apply volatility adjustment
        adjusted_limits['max_position_size'] *= self.volatility_adjustment
        adjusted_limits['max_portfolio_risk'] *= self.volatility_adjustment
        
        self.base_manager.update_risk_limits(adjusted_limits)
    
    def calculate_dynamic_position_size(self, symbol, signal_strength, market_conditions):
        """
        Calculate position size with dynamic adjustments
        
        Args:
            symbol: Trading symbol
            signal_strength: Strength of trading signal (0-1)
            market_conditions: Current market conditions
        
        Returns:
            float: Adjusted position size multiplier
        """
        base_size = signal_strength