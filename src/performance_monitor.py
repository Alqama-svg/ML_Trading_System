"""
Performance Monitoring System
Real-time performance tracking and analytics
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(returns, benchmark_returns=None, risk_free_rate=0.02):
    """
    Calculate comprehensive performance metrics
    
    Args:
        returns: Series of strategy returns
        benchmark_returns: Series of benchmark returns (optional)
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        dict: Performance metrics
    """
    if len(returns) == 0:
        return _empty_metrics()
    
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (1 + returns).prod() - 1
    metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
    metrics['volatility'] = returns.std() * np.sqrt(252)
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
    
    # Risk metrics
    metrics['max_drawdown'] = calculate_max_drawdown(returns)
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean() if len(returns[returns <= metrics['var_95']]) > 0 else metrics['var_95']
    
    # Trade statistics
    metrics['win_rate'] = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
    metrics['avg_win'] = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
    metrics['avg_loss'] = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
    metrics['profit_factor'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
    
    # Additional metrics
    metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else np.inf
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate)
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)
    
    # Benchmark comparison
    if benchmark_returns is not None:
        metrics.update(calculate_benchmark_metrics(returns, benchmark_returns))
    
    return metrics

def _empty_metrics():
    """Return empty metrics dictionary"""
    return {
        'total_return': 0,
        'annualized_return': 0,
        'volatility': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
        'var_95': 0,
        'cvar_95': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_factor': 0,
        'calmar_ratio': 0,
        'sortino_ratio': 0,
        'skewness': 0,
        'kurtosis': 0
    }

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino ratio"""
    if len(returns) == 0:
        return 0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_deviation = downside_returns.std() * np.sqrt(252)
    
    if downside_deviation == 0:
        return np.inf
    
    return excess_returns.mean() * np.sqrt(252) / downside_deviation

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    if len(returns) == 0:
        return 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()

def calculate_benchmark_metrics(returns, benchmark_returns):
    """Calculate benchmark comparison metrics"""
    # Align returns
    common_dates = returns.index.intersection(benchmark_returns.index)
    if len(common_dates) == 0:
        return {}
    
    strategy_aligned = returns.loc[common_dates]
    benchmark_aligned = benchmark_returns.loc[common_dates]
    
    # Calculate metrics
    excess_returns = strategy_aligned - benchmark_aligned
    
    metrics = {
        'alpha': excess_returns.mean() * 252,
        'beta': calculate_beta(strategy_aligned, benchmark_aligned),
        'information_ratio': excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0,
        'tracking_error': excess_returns.std() * np.sqrt(252),
        'correlation': strategy_aligned.corr(benchmark_aligned)
    }
    
    return metrics

def calculate_beta(returns, benchmark_returns):
    """Calculate beta coefficient"""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 1.0
    
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    return covariance / benchmark_variance if benchmark_variance > 0 else 1.0

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self):
        self.equity_curve = pd.Series(dtype=float)
        self.returns = pd.Series(dtype=float)
        self.trades = []
        self.daily_pnl = pd.Series(dtype=float)
        self.positions_history = []
        self.benchmark_returns = pd.Series(dtype=float)
        
    def update_portfolio_value(self, timestamp, portfolio_value):
        """Update portfolio value and calculate returns"""
        self.equity_curve[timestamp] = portfolio_value
        
        if len(self.equity_curve) > 1:
            previous_value = self.equity_curve.iloc[-2]
            if previous_value > 0:
                daily_return = (portfolio_value - previous_value) / previous_value
                self.returns[timestamp] = daily_return
                self.daily_pnl[timestamp] = portfolio_value - previous_value
    
    def add_trade(self, trade_info):
        """Add completed trade information"""
        required_fields = ['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'quantity', 'pnl']
        
        if all(field in trade_info for field in required_fields):
            # Calculate additional trade metrics
            trade_info['return'] = trade_info['pnl'] / (trade_info['entry_price'] * abs(trade_info['quantity']))
            trade_info['hold_period'] = (trade_info['exit_time'] - trade_info['entry_time']).days
            
            self.trades.append(trade_info)
    
    def update_positions(self, timestamp, positions):
        """Update current positions"""
        self.positions_history.append({
            'timestamp': timestamp,
            'positions': positions.copy()
        })
        
        # Keep only recent history
        if len(self.positions_history) > 1000:
            self.positions_history = self.positions_history[-1000:]
    
    def get_current_metrics(self):
        """Get current performance metrics"""
        if len(self.returns) == 0:
            return _empty_metrics()
        
        return calculate_metrics(self.returns, self.benchmark_returns)
    
    def get_trade_analysis(self):
        """Analyze completed trades"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_hold_period': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic statistics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        analysis = {
            'total_trades': len(trades_df),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
            'avg_win': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'largest_win': trades_df['pnl'].max(),
            'largest_loss': trades_df['pnl'].min(),
            'avg_hold_period': trades_df['hold_period'].mean(),
            'total_pnl': trades_df['pnl'].sum(),
            'gross_profit': winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0,
            'gross_loss': losing_trades['pnl'].sum() if len(losing_trades) > 0 else 0
        }
        
        # Profit factor
        analysis['profit_factor'] = abs(analysis['gross_profit'] / analysis['gross_loss']) if analysis['gross_loss'] != 0 else np.inf
        
        # Return statistics
        analysis['avg_return_per_trade'] = trades_df['return'].mean()
        analysis['std_return_per_trade'] = trades_df['return'].std()
        
        return analysis
    
    def get_rolling_metrics(self, window=30):
        """Get rolling performance metrics"""
        if len(self.returns) < window:
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=self.returns.index[window-1:])
        
        for i in range(window-1, len(self.returns)):
            window_returns = self.returns.iloc[i-window+1:i+1]
            
            rolling_metrics.loc[self.returns.index[i], 'rolling_return'] = (1 + window_returns).prod() - 1
            rolling_metrics.loc[self.returns.index[i], 'rolling_volatility'] = window_returns.std() * np.sqrt(252)
            rolling_metrics.loc[self.returns.index[i], 'rolling_sharpe'] = calculate_sharpe_ratio(window_returns)
            rolling_metrics.loc[self.returns.index[i], 'rolling_max_dd'] = calculate_max_drawdown(window_returns)
        
        return rolling_metrics
    
    def get_monthly_returns(self):
        """Get monthly return breakdown"""
        if len(self.returns) == 0:
            return pd.DataFrame()
        
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create monthly returns table
        monthly_table = monthly_returns.to_frame('return')
        monthly_table['year'] = monthly_table.index.year
        monthly_table['month'] = monthly_table.index.month
        
        # Pivot to create calendar view
        pivot_table = monthly_table.pivot(index='year', columns='month', values='return')
        
        # Add yearly totals
        yearly_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        pivot_table['Year'] = yearly_returns.reindex(pivot_table.index)
        
        return pivot_table
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        current_metrics = self.get_current_metrics()
        trade_analysis = self.get_trade_analysis()
        
        report = {
            'summary': {
                'total_return': current_metrics['total_return'],
                'annualized_return': current_metrics['annualized_return'],
                'volatility': current_metrics['volatility'],
                'sharpe_ratio': current_metrics['sharpe_ratio'],
                'max_drawdown': current_metrics['max_drawdown'],
                'calmar_ratio': current_metrics['calmar_ratio']
            },
            'trade_statistics': trade_analysis,
            'risk_metrics': {
                'var_95': current_metrics['var_95'],
                'cvar_95': current_metrics['cvar_95'],
                'skewness': current_metrics['skewness'],
                'kurtosis': current_metrics['kurtosis'],
                'sortino_ratio': current_metrics['sortino_ratio']
            }
        }
        
        # Add benchmark comparison if available
        if len(self.benchmark_returns) > 0:
            benchmark_metrics = calculate_benchmark_metrics(self.returns, self.benchmark_returns)
            report['benchmark_comparison'] = benchmark_metrics
        
        return report

class RealTimeAnalytics:
    """Real-time analytics and alerts"""
    
    def __init__(self, performance_monitor):
        self.monitor = performance_monitor
        self.alerts = []
        self.thresholds = {
            'max_daily_loss': -0.05,  # 5% daily loss
            'max_drawdown': -0.20,    # 20% drawdown
            'min_sharpe': 0.5,        # Minimum Sharpe ratio
            'max_volatility': 0.30    # Maximum volatility
        }
    
    def check_performance_alerts(self):
        """Check for performance-based alerts"""
        alerts = []
        
        if len(self.monitor.returns) == 0:
            return alerts
        
        current_metrics = self.monitor.get_current_metrics()
        
        # Daily loss alert
        if len(self.monitor.daily_pnl) > 0:
            latest_pnl = self.monitor.daily_pnl.iloc[-1]
            if latest_pnl < self.thresholds['max_daily_loss'] * self.monitor.equity_curve.iloc[-1]:
                alerts.append({
                    'type': 'DAILY_LOSS',
                    'message': f"Daily loss of {latest_pnl:.2f} exceeds threshold",
                    'severity': 'HIGH'
                })
        
        # Drawdown alert
        if current_metrics['max_drawdown'] < self.thresholds['max_drawdown']:
            alerts.append({
                'type': 'MAX_DRAWDOWN',
                'message': f"Drawdown of {current_metrics['max_drawdown']:.2%} exceeds threshold",
                'severity': 'HIGH'
            })
        
        # Sharpe ratio alert
        if current_metrics['sharpe_ratio'] < self.thresholds['min_sharpe']:
            alerts.append({
                'type': 'LOW_SHARPE',
                'message': f"Sharpe ratio of {current_metrics['sharpe_ratio']:.2f} below threshold",
                'severity': 'MEDIUM'
            })
        
        # Volatility alert
        if current_metrics['volatility'] > self.thresholds['max_volatility']:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'message': f"Volatility of {current_metrics['volatility']:.2%} exceeds threshold",
                'severity': 'MEDIUM'
            })
        
        self.alerts.extend(alerts)
        return alerts
    
    def update_thresholds(self, new_thresholds):
        """Update alert thresholds"""
        self.thresholds.update(new_thresholds)
    
    def get_performance_attribution(self, positions_data, returns_data):
        """Calculate performance attribution by position"""
        if not positions_data or returns_data.empty:
            return {}
        
        attribution = {}
        
        for position_record in positions_data[-30:]:  # Last 30 records
            timestamp = position_record['timestamp']
            positions = position_record['positions']
            
            for symbol, shares in positions.items():
                if symbol in returns_data.columns and timestamp in returns_data.index:
                    asset_return = returns_data.loc[timestamp, symbol]
                    contribution = shares * asset_return
                    
                    if symbol not in attribution:
                        attribution[symbol] = []
                    attribution[symbol].append(contribution)
        
        # Aggregate attribution
        final_attribution = {}
        for symbol, contributions in attribution.items():
            final_attribution[symbol] = {
                'total_contribution': sum(contributions),
                'avg_contribution': np.mean(contributions),
                'contribution_volatility': np.std(contributions)
            }
        
        return final_attribution

class StrategyComparison:
    """Compare multiple strategies"""
    
    def __init__(self):
        self.strategies = {}
    
    def add_strategy(self, name, returns, metadata=None):
        """Add strategy for comparison"""
        self.strategies[name] = {
            'returns': returns,
            'metadata': metadata or {},
            'metrics': calculate_metrics(returns)
        }
    
    def compare_strategies(self):
        """Compare all strategies"""
        if not self.strategies:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, strategy in self.strategies.items():
            metrics = strategy['metrics']
            comparison_data.append({
                'Strategy': name,
                'Total Return': metrics['total_return'],
                'Annualized Return': metrics['annualized_return'],
                'Volatility': metrics['volatility'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Calmar Ratio': metrics['calmar_ratio'],
                'Win Rate': metrics['win_rate'],
                'Profit Factor': metrics['profit_factor']
            })
        
        return pd.DataFrame(comparison_data).set_index('Strategy')
    
    def rank_strategies(self, metric='sharpe_ratio'):
        """Rank strategies by specific metric"""
        if not self.strategies:
            return []
        
        strategy_scores = []
        for name, strategy in self.strategies.items():
            score = strategy['metrics'].get(metric, 0)
            strategy_scores.append((name, score))
        
        # Sort by score (descending for most metrics)
        reverse = metric not in ['max_drawdown', 'volatility']  # These should be minimized
        strategy_scores.sort(key=lambda x: x[1], reverse=reverse)
        
        return strategy_scores

class BacktestAnalyzer:
    """Detailed backtest analysis"""
    
    def __init__(self, returns, trades=None, positions=None):
        self.returns = returns
        self.trades = trades or []
        self.positions = positions or []
        
    def analyze_returns_distribution(self):
        """Analyze return distribution characteristics"""
        if len(self.returns) == 0:
            return {}
        
        analysis = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'jarque_bera_test': stats.jarque_bera(self.returns),
            'shapiro_wilk_test': stats.shapiro(self.returns) if len(self.returns) <= 5000 else None
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            analysis[f'percentile_{p}'] = np.percentile(self.returns, p)
        
        return analysis
    
    def calculate_rolling_statistics(self, windows=[30, 60, 252]):
        """Calculate rolling statistics"""
        rolling_stats = {}
        
        for window in windows:
            if len(self.returns) >= window:
                rolling_stats[f'rolling_{window}d'] = {
                    'return': self.returns.rolling(window).apply(lambda x: (1 + x).prod() - 1),
                    'volatility': self.returns.rolling(window).std() * np.sqrt(252),
                    'sharpe': self.returns.rolling(window).apply(lambda x: calculate_sharpe_ratio(x)),
                    'max_drawdown': self.returns.rolling(window).apply(lambda x: calculate_max_drawdown(x))
                }
        
        return rolling_stats
    
    def identify_regime_periods(self, volatility_threshold=0.02):
        """Identify different market regime periods"""
        if len(self.returns) < 30:
            return []
        
        # Calculate rolling volatility
        rolling_vol = self.returns.rolling(30).std()
        
        regimes = []
        current_regime = None
        regime_start = None
        
        for date, vol in rolling_vol.items():
            if pd.isna(vol):
                continue
                
            if vol > volatility_threshold:
                new_regime = 'high_vol'
            else:
                new_regime = 'low_vol'
            
            if current_regime != new_regime:
                if current_regime is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start': regime_start,
                        'end': date,
                        'duration': (date - regime_start).days,
                        'avg_return': self.returns[regime_start:date].mean(),
                        'volatility': self.returns[regime_start:date].std()
                    })
                
                current_regime = new_regime
                regime_start = date
        
        # Add final regime
        if current_regime is not None and regime_start is not None:
            regimes.append({
                'regime': current_regime,
                'start': regime_start,
                'end': self.returns.index[-1],
                'duration': (self.returns.index[-1] - regime_start).days,
                'avg_return': self.returns[regime_start:].mean(),
                'volatility': self.returns[regime_start:].std()
            })
        
        return regimes
    
    def calculate_drawdown_analysis(self):
        """Detailed drawdown analysis"""
        if len(self.returns) == 0:
            return {}
        
        # Calculate cumulative returns and running max
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Find all drawdown periods
        drawdown_periods = []
        in_drawdown = False
        drawdown_start = None
        
        for date, dd in drawdown.items():
            if dd < -0.001 and not in_drawdown:  # Start of drawdown (>0.1%)
                in_drawdown = True
                drawdown_start = date
            elif dd >= -0.001 and in_drawdown:  # End of drawdown
                in_drawdown = False
                if drawdown_start is not None:
                    period_dd = drawdown[drawdown_start:date]
                    drawdown_periods.append({
                        'start': drawdown_start,
                        'end': date,
                        'duration': (date - drawdown_start).days,
                        'max_drawdown': period_dd.min(),
                        'recovery_time': (date - drawdown_start).days
                    })
        
        # Summary statistics
        if drawdown_periods:
            analysis = {
                'total_drawdown_periods': len(drawdown_periods),
                'avg_drawdown_duration': np.mean([dd['duration'] for dd in drawdown_periods]),
                'max_drawdown_duration': max([dd['duration'] for dd in drawdown_periods]),
                'avg_drawdown_depth': np.mean([dd['max_drawdown'] for dd in drawdown_periods]),
                'max_drawdown_depth': min([dd['max_drawdown'] for dd in drawdown_periods]),
                'drawdown_periods': drawdown_periods
            }
        else:
            analysis = {
                'total_drawdown_periods': 0,
                'avg_drawdown_duration': 0,
                'max_drawdown_duration': 0,
                'avg_drawdown_depth': 0,
                'max_drawdown_depth': 0,
                'drawdown_periods': []
            }
        
        return analysis

class LiveTradingMonitor:
    """Monitor live trading performance"""
    
    def __init__(self):
        self.live_metrics = {}
        self.alerts = []
        self.daily_targets = {}
        
    def set_daily_targets(self, targets):
        """Set daily performance targets"""
        self.daily_targets = targets
    
    def update_live_metrics(self, timestamp, portfolio_value, positions, market_data):
        """Update live trading metrics"""
        self.live_metrics[timestamp] = {
            'portfolio_value': portfolio_value,
            'positions': positions.copy(),
            'market_data': market_data.copy()
        }
        
        # Check targets
        self._check_daily_targets(timestamp, portfolio_value)
    
    def _check_daily_targets(self, timestamp, portfolio_value):
        """Check if daily targets are being met"""
        today = timestamp.date()
        
        if today not in self.daily_targets:
            return
        
        targets = self.daily_targets[today]
        
        # Calculate today's performance
        today_metrics = [m for t, m in self.live_metrics.items() if t.date() == today]
        
        if len(today_metrics) < 2:
            return
        
        start_value = today_metrics[0]['portfolio_value']
        current_value = today_metrics[-1]['portfolio_value']
        daily_return = (current_value - start_value) / start_value
        
        # Check targets
        if 'min_return' in targets and daily_return < targets['min_return']:
            self.alerts.append({
                'type': 'TARGET_MISS',
                'message': f"Daily return {daily_return:.2%} below target {targets['min_return']:.2%}",
                'timestamp': timestamp
            })
        
        if 'max_loss' in targets and daily_return < targets['max_loss']:
            self.alerts.append({
                'type': 'LOSS_LIMIT',
                'message': f"Daily loss {daily_return:.2%} exceeds limit {targets['max_loss']:.2%}",
                'timestamp': timestamp
            })
    
    def get_intraday_metrics(self, date=None):
        """Get intraday performance metrics"""
        if date is None:
            date = pd.Timestamp.now().date()
        
        day_metrics = [(t, m) for t, m in self.live_metrics.items() if t.date() == date]
        
        if len(day_metrics) < 2:
            return {}
        
        # Calculate intraday returns
        values = [m['portfolio_value'] for t, m in day_metrics]
        times = [t for t, m in day_metrics]
        
        returns = pd.Series(values, index=times).pct_change().dropna()
        
        return {
            'intraday_return': (values[-1] - values[0]) / values[0],
            'intraday_high': max(values),
            'intraday_low': min(values),
            'intraday_volatility': returns.std() * np.sqrt(len(returns)) if len(returns) > 1 else 0,
            'number_of_updates': len(day_metrics)
        }

# Utility functions for performance analysis

def calculate_information_coefficient(predictions, actual_returns):
    """Calculate information coefficient (IC)"""
    if len(predictions) != len(actual_returns) or len(predictions) < 10:
        return 0
    
    # Remove NaN values
    data = pd.DataFrame({'pred': predictions, 'actual': actual_returns}).dropna()
    
    if len(data) < 10:
        return 0
    
    return data['pred'].corr(data['actual'])

def calculate_hit_rate(predictions, actual_returns, threshold=0):
    """Calculate hit rate (percentage of correct direction predictions)"""
    if len(predictions) != len(actual_returns):
        return 0
    
    pred_direction = (predictions > threshold).astype(int)
    actual_direction = (actual_returns > threshold).astype(int)
    
    return (pred_direction == actual_direction).mean()

def calculate_capture_ratios(strategy_returns, benchmark_returns):
    """Calculate upside and downside capture ratios"""
    if len(strategy_returns) != len(benchmark_returns):
        return {'upside_capture': 0, 'downside_capture': 0}
    
    # Separate up and down periods
    up_periods = benchmark_returns > 0
    down_periods = benchmark_returns < 0
    
    if up_periods.sum() == 0 or down_periods.sum() == 0:
        return {'upside_capture': 0, 'downside_capture': 0}
    
    # Calculate capture ratios
    upside_capture = strategy_returns[up_periods].mean() / benchmark_returns[up_periods].mean()
    downside_capture = strategy_returns[down_periods].mean() / benchmark_returns[down_periods].mean()
    
    return {
        'upside_capture': upside_capture,
        'downside_capture': downside_capture
    }

def calculate_tail_ratios(returns):
    """Calculate tail ratios for risk assessment"""
    if len(returns) < 100:
        return {}
    
    # Calculate percentiles
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    p99 = np.percentile(returns, 99)
    p1 = np.percentile(returns, 1)
    
    return {
        'tail_ratio_95_5': abs(p95 / p5) if p5 != 0 else np.inf,
        'tail_ratio_99_1': abs(p99 / p1) if p1 != 0 else np.inf,
        'right_tail_mean': returns[returns > p95].mean(),
        'left_tail_mean': returns[returns < p5].mean()
<<<<<<< HEAD
    }
=======
    }
>>>>>>> f1909685739746bbe77927120694a7980b73754a
