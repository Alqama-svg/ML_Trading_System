"""
LLM Integration for Trading System
• Use ChatGPT API for strategy ideation
• Automated documentation generation
• Code review and optimization suggestions
• Market commentary generation
"""

import openai
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class LLMAssistant:
    def __init__(self, api_key=None):
        """
        Initialize LLM Assistant
        
        Args:
            api_key: OpenAI API key (can be set via environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: OpenAI API key not provided. LLM features will use mock responses.")
        
        self.conversation_history = []
        
    def generate_market_commentary(self, market_data, technical_indicators=None, news_sentiment=None):
        """
        Generate market commentary based on current market conditions
        
        Args:
            market_data: Dictionary containing current market data
            technical_indicators: Dictionary of technical indicator values
            news_sentiment: Current news sentiment score
        """
        try:
            # Prepare context
            context = self._prepare_market_context(market_data, technical_indicators, news_sentiment)
            
            prompt = f"""
            As a professional financial analyst, provide a concise market commentary based on the following data:
            
            {context}
            
            Please provide:
            1. Current market assessment (2-3 sentences)
            2. Key technical levels to watch
            3. Risk factors to consider
            4. Short-term outlook (next 1-5 trading days)
            
            Keep the analysis professional, objective, and actionable for traders.
            """
            
            if self.api_key:
                response = self._call_openai_api(prompt, max_tokens=500)
                return response
            else:
                return self._generate_mock_commentary(market_data)
                
        except Exception as e:
            print(f"Error generating market commentary: {e}")
            return self._generate_mock_commentary(market_data)
    
    def generate_strategy_ideas(self, market_conditions, risk_tolerance="medium", timeframe="daily"):
        """
        Generate trading strategy ideas based on market conditions
        
        Args:
            market_conditions: Dictionary describing current market conditions
            risk_tolerance: "low", "medium", or "high"
            timeframe: Trading timeframe ("intraday", "daily", "weekly")
        """
        try:
            prompt = f"""
            As a quantitative trading strategist, suggest 3 trading strategies for the following conditions:
            
            Market Conditions: {json.dumps(market_conditions, indent=2)}
            Risk Tolerance: {risk_tolerance}
            Timeframe: {timeframe}
            
            For each strategy, provide:
            1. Strategy name and type
            2. Entry/exit conditions
            3. Risk management approach
            4. Expected performance characteristics
            5. Market conditions where it works best
            
            Focus on strategies that can be systematically implemented and backtested.
            """
            
            if self.api_key:
                response = self._call_openai_api(prompt, max_tokens=800)
                return self._parse_strategy_response(response)
            else:
                return self._generate_mock_strategies(market_conditions, risk_tolerance)
                
        except Exception as e:
            print(f"Error generating strategy ideas: {e}")
            return self._generate_mock_strategies(market_conditions, risk_tolerance)
    
    def review_trading_code(self, code, strategy_type="general"):
        """
        Review trading code and provide optimization suggestions
        
        Args:
            code: Trading strategy code to review
            strategy_type: Type of strategy being reviewed
        """
        try:
            prompt = f"""
            As a senior quantitative developer, review this {strategy_type} trading strategy code and provide feedback:
            
            ```python
            {code}
            ```
            
            Please analyze:
            1. Code quality and structure
            2. Potential bugs or issues
            3. Performance optimization opportunities
            4. Risk management improvements
            5. Best practices compliance
            
            Provide specific, actionable suggestions for improvement.
            """
            
            if self.api_key:
                response = self._call_openai_api(prompt, max_tokens=600)
                return response
            else:
                return self._generate_mock_code_review()
                
        except Exception as e:
            print(f"Error reviewing code: {e}")
            return self._generate_mock_code_review()
    
    def generate_documentation(self, strategy_name, strategy_description, parameters):
        """
        Generate comprehensive documentation for a trading strategy
        
        Args:
            strategy_name: Name of the trading strategy
            strategy_description: Brief description of the strategy
            parameters: Dictionary of strategy parameters
        """
        try:
            prompt = f"""
            Generate comprehensive documentation for a trading strategy with the following details:
            
            Strategy Name: {strategy_name}
            Description: {strategy_description}
            Parameters: {json.dumps(parameters, indent=2)}
            
            Create documentation including:
            1. Executive Summary
            2. Strategy Logic and Methodology
            3. Parameter Descriptions
            4. Risk Considerations
            5. Implementation Notes
            6. Performance Expectations
            7. Market Conditions Suitability
            
            Format as professional strategy documentation suitable for institutional use.
            """
            
            if self.api_key:
                response = self._call_openai_api(prompt, max_tokens=1000)
                return response
            else:
                return self._generate_mock_documentation(strategy_name)
                
        except Exception as e:
            print(f"Error generating documentation: {e}")
            return self._generate_mock_documentation(strategy_name)
    
    def analyze_market_regime(self, price_data, volume_data=None, volatility_data=None):
        """
        Analyze current market regime using LLM
        
        Args:
            price_data: Recent price data
            volume_data: Recent volume data (optional)
            volatility_data: Recent volatility data (optional)
        """
        try:
            # Calculate basic statistics
            price_change = (price_data[-1] / price_data[0] - 1) * 100
            volatility = np.std(np.diff(price_data) / price_data[:-1]) * 100
            
            trend_direction = "upward" if price_change > 2 else "downward" if price_change < -2 else "sideways"
            vol_regime = "high" if volatility > 2 else "low" if volatility < 0.5 else "normal"
            
            prompt = f"""
            Analyze the current market regime based on these characteristics:
            
            Recent Performance: {price_change:.2f}% over the analysis period
            Trend Direction: {trend_direction}
            Volatility Regime: {vol_regime} (realized vol: {volatility:.2f}%)
            
            Determine the market regime and provide:
            1. Market regime classification (trending/mean-reverting/volatile/calm)
            2. Recommended trading approach for this regime
            3. Strategies to avoid in this environment
            4. Key indicators to monitor for regime changes
            
            Provide actionable insights for systematic trading.
            """
            
            if self.api_key:
                response = self._call_openai_api(prompt, max_tokens=400)
                return response
            else:
                return self._generate_mock_regime_analysis(trend_direction, vol_regime)
                
        except Exception as e:
            print(f"Error analyzing market regime: {e}")
            return self._generate_mock_regime_analysis("sideways", "normal")
    
    def _call_openai_api(self, prompt, max_tokens=500, temperature=0.7):
        """Call OpenAI API with error handling"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional quantitative trading analyst with expertise in financial markets, technical analysis, and algorithmic trading strategies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "API call failed. Using fallback response."
    
    def _prepare_market_context(self, market_data, technical_indicators, news_sentiment):
        """Prepare market context for LLM analysis"""
        context = []
        
        # Market data
        if market_data:
            context.append("Current Market Data:")
            for key, value in market_data.items():
                context.append(f"- {key}: {value}")
        
        # Technical indicators
        if technical_indicators:
            context.append("\nTechnical Indicators:")
            for key, value in technical_indicators.items():
                if isinstance(value, (int, float)):
                    context.append(f"- {key}: {value:.2f}")
                else:
                    context.append(f"- {key}: {value}")
        
        # News sentiment
        if news_sentiment is not None:
            sentiment_desc = "positive" if news_sentiment > 0.1 else "negative" if news_sentiment < -0.1 else "neutral"
            context.append(f"\nNews Sentiment: {sentiment_desc} ({news_sentiment:.2f})")
        
        return "\n".join(context)
    
    def _parse_strategy_response(self, response):
        """Parse strategy response into structured format"""
        # Simple parsing - in production, use more sophisticated NLP
        strategies = []
        lines = response.split('\n')
        
        current_strategy = {}
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '**Strategy', 'Strategy')):
                if current_strategy:
                    strategies.append(current_strategy)
                current_strategy = {'description': line}
            elif line and current_strategy:
                if 'details' not in current_strategy:
                    current_strategy['details'] = []
                current_strategy['details'].append(line)
        
        if current_strategy:
            strategies.append(current_strategy)
        
        return strategies
    
    # Mock response generators for when API is not available
    def _generate_mock_commentary(self, market_data):
        """Generate mock market commentary"""
        price = market_data.get('current_price', 100)
        
        return f"""
        **Market Assessment:** The current price level of ${price:.2f} shows mixed signals with moderate volatility. 
        Recent trading patterns suggest consolidation phase with potential for breakout.
        
        **Key Technical Levels:** 
        - Support: ${price * 0.98:.2f}
        - Resistance: ${price * 1.02:.2f}
        
        **Risk Factors:** Monitor volume patterns and broader market sentiment for direction confirmation.
        
        **Short-term Outlook:** Expect continued sideways movement with bias toward the prevailing trend.
        Range-bound trading likely over next 1-3 sessions.
        """
    
    def _generate_mock_strategies(self, market_conditions, risk_tolerance):
        """Generate mock strategy suggestions"""
        return [
            {
                'description': 'Mean Reversion Strategy',
                'details': [
                    'Entry: Price moves 2+ standard deviations from mean',
                    'Exit: Return to mean or 1 standard deviation',
                    'Risk: 2% position sizing with stop-loss at 3 standard deviations'
                ]
            },
            {
                'description': 'Momentum Breakout Strategy',
                'details': [
                    'Entry: Break above resistance with increased volume',
                    'Exit: Moving average crossover or volume decline',
                    'Risk: Trailing stop-loss at 1.5% below entry'
                ]
            },
            {
                'description': 'Volatility Contraction Strategy',
                'details': [
                    'Entry: Low volatility followed by volume spike',
                    'Exit: Volatility expansion beyond 20-day average',
                    'Risk: Fixed percentage stop-loss based on ATR'
                ]
            }
        ]
    
    def _generate_mock_code_review(self):
        """Generate mock code review"""
        return """
        **Code Review Summary:**
        
        **Strengths:**
        - Clear variable naming and structure
        - Proper error handling in data processing
        - Good separation of concerns
        
        **Areas for Improvement:**
        1. Add input validation for edge cases
        2. Implement proper logging for debugging
        3. Consider vectorization for performance optimization
        4. Add unit tests for critical functions
        
        **Risk Management:**
        - Include position sizing limits
        - Add maximum drawdown controls
        - Implement timeout mechanisms for stuck positions
        
        **Performance Optimization:**
        - Cache frequently used calculations
        - Use pandas operations instead of loops where possible
        - Consider parallel processing for backtesting
        """
    
    def _generate_mock_documentation(self, strategy_name):
        """Generate mock strategy documentation"""
        return f"""
        # {strategy_name} Strategy Documentation
        
        ## Executive Summary
        {strategy_name} is a systematic trading strategy designed to capture market inefficiencies 
        through quantitative analysis and risk-controlled execution.
        
        ## Strategy Logic
        The strategy employs technical indicators and statistical models to identify 
        optimal entry and exit points while maintaining strict risk management protocols.
        
        ## Key Parameters
        - Entry threshold: Statistical significance level
        - Exit conditions: Profit target and stop-loss levels
        - Position sizing: Risk-based allocation
        
        ## Risk Considerations
        - Maximum portfolio exposure limits
        - Drawdown controls and circuit breakers
        - Market regime adaptability
        
        ## Implementation Notes
        Strategy requires real-time data feeds and low-latency execution capabilities.
        Regular performance monitoring and parameter optimization recommended.
        """
    
    def _generate_mock_regime_analysis(self, trend_direction, vol_regime):
        """Generate mock market regime analysis"""
        return f"""
        **Market Regime Analysis:**
        
        **Current Classification:** {trend_direction.title()} trend with {vol_regime} volatility
        
        **Recommended Approach:**
        - For {trend_direction} trends: Use momentum-based strategies
        - In {vol_regime} volatility: Adjust position sizing accordingly
        
        **Strategies to Avoid:**
        - Mean reversion in strong trending markets
        - High-frequency strategies in low volatility periods
        
        **Regime Change Indicators:**
        - Volume pattern shifts
        - Volatility breakouts/breakdowns
        - Cross-asset correlation changes
        
        **Tactical Recommendations:**
        Monitor for regime transitions and adjust strategy allocation dynamically.
        """

class StrategyIdeationEngine:
    """
    Advanced strategy ideation using LLM capabilities
    """
    
    def __init__(self, llm_assistant):
        self.llm = llm_assistant
        self.strategy_database = []
        
    def brainstorm_strategies(self, market_data, constraints=None):
        """
        Brainstorm new trading strategies based on market conditions
        
        Args:
            market_data: Current market data and conditions
            constraints: Trading constraints (capital, instruments, etc.)
        """
        # Analyze current market patterns
        patterns = self._identify_market_patterns(market_data)
        
        # Generate strategy ideas
        strategy_prompt = f"""
        Based on these market patterns: {patterns}
        And constraints: {constraints or 'None specified'}
        
        Generate 5 innovative trading strategy concepts that could exploit current market conditions.
        For each strategy, provide:
        1. Core hypothesis
        2. Implementation approach
        3. Risk/reward profile
        4. Market conditions where it thrives
        """
        
        ideas = self.llm._call_openai_api(strategy_prompt, max_tokens=800)
        return self._structure_strategy_ideas(ideas)
    
    def _identify_market_patterns(self, market_data):
        """Identify key patterns in market data"""
        patterns = []
        
        if 'volatility' in market_data:
            vol = market_data['volatility']
            if vol > 0.25:
                patterns.append("High volatility environment")
            elif vol < 0.10:
                patterns.append("Low volatility environment")
        
        if 'correlation' in market_data:
            corr = market_data['correlation']
            if corr > 0.8:
                patterns.append("High cross-asset correlation")
            elif corr < 0.3:
                patterns.append("Low cross-asset correlation")
        
        if 'trend_strength' in market_data:
            trend = market_data['trend_strength']
            if trend > 0.7:
                patterns.append("Strong trending market")
            elif trend < 0.3:
                patterns.append("Range-bound market")
        
        return ", ".join(patterns) if patterns else "Mixed market conditions"
    
    def _structure_strategy_ideas(self, ideas_text):
        """Structure strategy ideas into organized format"""
        # Simple parsing - can be enhanced with better NLP
        ideas = []
        sections = ideas_text.split('\n\n')
        
        for section in sections:
            if any(keyword in section.lower() for keyword in ['strategy', 'approach', 'concept']):
                ideas.append({
                    'raw_text': section.strip(),
                    'timestamp': datetime.now(),
                    'status': 'new_idea'
                })
        
        return ideas

class AutomatedReporting:
    """
    Automated report generation using LLM
    """
    
    def __init__(self, llm_assistant):
        self.llm = llm_assistant
        
    def generate_daily_report(self, portfolio_data, market_data, performance_metrics):
        """
        Generate daily trading report
        
        Args:
            portfolio_data: Current portfolio positions and values
            market_data: Market performance data
            performance_metrics: Portfolio performance metrics
        """
        report_prompt = f"""
        Generate a professional daily trading report with the following data:
        
        Portfolio Summary:
        {json.dumps(portfolio_data, indent=2)}
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Performance Metrics:
        {json.dumps(performance_metrics, indent=2)}
        
        Include sections for:
        1. Executive Summary
        2. Portfolio Performance
        3. Market Commentary
        4. Risk Assessment
        5. Tomorrow's Outlook
        
        Keep it concise but comprehensive for stakeholder review.
        """
        
        return self.llm._call_openai_api(report_prompt, max_tokens=1200)
    
    def generate_strategy_performance_report(self, strategy_name, backtest_results):
        """
        Generate strategy performance analysis report
        
        Args:
            strategy_name: Name of the strategy
            backtest_results: Dictionary containing backtest metrics
        """
        report_prompt = f"""
        Create a comprehensive performance analysis report for {strategy_name} with these results:
        
        {json.dumps(backtest_results, indent=2)}
        
        Analyze:
        1. Overall Performance Assessment
        2. Risk-Adjusted Returns
        3. Drawdown Analysis
        4. Strategy Strengths and Weaknesses
        5. Recommended Improvements
        6. Market Condition Sensitivity
        
        Provide actionable insights for strategy optimization.
        """
        
        return self.llm._call_openai_api(report_prompt, max_tokens=1000)

class LLMBacktestAnalyzer:
    """
    LLM-powered backtest analysis and interpretation
    """
    
    def __init__(self, llm_assistant):
        self.llm = llm_assistant
        
    def analyze_backtest_results(self, results, benchmark_comparison=None):
        """
        Analyze backtest results using LLM
        
        Args:
            results: Backtest results dictionary
            benchmark_comparison: Comparison with benchmark performance
        """
        analysis_prompt = f"""
        Analyze these trading strategy backtest results as a quantitative analyst:
        
        Results:
        {json.dumps(results, indent=2)}
        
        Benchmark Comparison:
        {json.dumps(benchmark_comparison, indent=2) if benchmark_comparison else 'None provided'}
        
        Provide analysis on:
        1. Performance quality assessment
        2. Risk characteristics evaluation
        3. Strategy robustness indicators
        4. Potential overfitting concerns
        5. Recommendations for live trading
        6. Parameter sensitivity analysis
        
        Focus on practical insights for strategy deployment.
        """
        
        return self.llm._call_openai_api(analysis_prompt, max_tokens=800)
    
    def identify_performance_drivers(self, equity_curve, trade_analysis):
        """
        Identify key drivers of strategy performance
        
        Args:
            equity_curve: Time series of portfolio values
            trade_analysis: Detailed trade-by-trade analysis
        """
        driver_prompt = f"""
        Identify the key performance drivers for this trading strategy:
        
        Trade Analysis Summary:
        - Total trades: {trade_analysis.get('total_trades', 'N/A')}
        - Win rate: {trade_analysis.get('win_rate', 'N/A')}
        - Average win: {trade_analysis.get('avg_win', 'N/A')}
        - Average loss: {trade_analysis.get('avg_loss', 'N/A')}
        - Profit factor: {trade_analysis.get('profit_factor', 'N/A')}
        
        Analyze what drives performance:
        1. Trade frequency and timing patterns
        2. Win/loss distribution characteristics
        3. Market condition dependencies
        4. Position sizing effectiveness
        5. Entry/exit timing quality
        
        Provide actionable insights for optimization.
        """
        
        return self.llm._call_openai_api(driver_prompt, max_tokens=600)

class MarketNewsAnalyzer:
    """
    News and sentiment analysis using LLM
    """
    
    def __init__(self, llm_assistant):
        self.llm = llm_assistant
        
    def analyze_news_impact(self, news_headlines, market_data_before, market_data_after):
        """
        Analyze news impact on market movements
        
        Args:
            news_headlines: List of news headlines
            market_data_before: Market data before news
            market_data_after: Market data after news
        """
        impact_prompt = f"""
        Analyze the market impact of these news events:
        
        News Headlines:
        {chr(10).join([f"- {headline}" for headline in news_headlines[:10]])}
        
        Market Before: {json.dumps(market_data_before, indent=2)}
        Market After: {json.dumps(market_data_after, indent=2)}
        
        Assess:
        1. News sentiment and likely market direction
        2. Actual vs expected market reaction
        3. Persistence of news impact
        4. Trading opportunities identified
        5. Risk factors highlighted by news
        
        Provide insights for news-driven trading strategies.
        """
        
        return self.llm._call_openai_api(impact_prompt, max_tokens=700)
    
    def generate_sentiment_score(self, text_data):
        """
        Generate sentiment scores for text data
        
        Args:
            text_data: List of text strings to analyze
        """
        sentiment_prompt = f"""
        Analyze the sentiment of these financial texts and provide scores from -1 (very negative) to +1 (very positive):
        
        Texts to analyze:
        {chr(10).join([f"{i+1}. {text[:200]}..." for i, text in enumerate(text_data[:5])])}
        
        For each text, provide:
        1. Sentiment score (-1 to +1)
        2. Key sentiment drivers
        3. Market relevance (1-10)
        4. Confidence level (1-10)
        
        Format as JSON for easy parsing.
        """
        
        response = self.llm._call_openai_api(sentiment_prompt, max_tokens=400)
        return self._parse_sentiment_response(response)
    
    def _parse_sentiment_response(self, response):
        """Parse sentiment analysis response"""
        try:
            # Simple parsing - in production, use more robust JSON parsing
            lines = response.split('\n')
            sentiments = []
            
            for line in lines:
                if 'score' in line.lower() and any(char.isdigit() or char in '.-' for char in line):
                    # Extract numeric sentiment score
                    import re
                    scores = re.findall(r'-?\d*\.?\d+', line)
                    if scores:
                        score = float(scores[0])
                        sentiments.append(max(-1, min(1, score)))  # Clamp to [-1, 1]
            
            return sentiments if sentiments else [0] * 5  # Default neutral scores
        except:
            return [0] * 5  # Default neutral scores

class StrategyOptimizer:
    """
    LLM-assisted strategy optimization
    """
    
    def __init__(self, llm_assistant):
        self.llm = llm_assistant
        
    def suggest_parameter_optimization(self, strategy_description, current_params, performance_history):
        """
        Suggest parameter optimizations based on performance
        
        Args:
            strategy_description: Description of the trading strategy
            current_params: Current parameter values
            performance_history: Historical performance data
        """
        optimization_prompt = f"""
        Suggest parameter optimizations for this trading strategy:
        
        Strategy: {strategy_description}
        Current Parameters: {json.dumps(current_params, indent=2)}
        Performance History: {json.dumps(performance_history, indent=2)}
        
        Recommend:
        1. Parameters most likely to improve performance
        2. Suggested value ranges for testing
        3. Parameters to avoid changing (stable ones)
        4. Optimization methodology approach
        5. Overfitting risk assessment
        
        Focus on robust, out-of-sample improvements.
        """
        
        return self.llm._call_openai_api(optimization_prompt, max_tokens=600)
    
    def analyze_parameter_sensitivity(self, param_sweep_results):
        """
        Analyze parameter sensitivity results
        
        Args:
            param_sweep_results: Results from parameter sweep analysis
        """
        sensitivity_prompt = f"""
        Analyze parameter sensitivity for strategy optimization:
        
        Parameter Sweep Results:
        {json.dumps(param_sweep_results, indent=2)}
        
        Identify:
        1. Most sensitive parameters (high impact on performance)
        2. Robust parameter ranges (stable performance)
        3. Parameter interactions and dependencies
        4. Recommended parameter bounds
        5. Overfitting warning signs
        
        Provide guidance for robust parameter selection.
        """
        
        return self.llm._call_openai_api(sensitivity_prompt, max_tokens=500)

# Utility functions for LLM integration

def setup_llm_environment():
    """Setup LLM environment with proper configurations"""
    config = {
        'max_tokens': 1000,
        'temperature': 0.7,
        'model': 'gpt-3.5-turbo',
        'timeout': 30,
        'retry_attempts': 3
    }
    return config

def validate_llm_response(response, expected_format=None):
    """Validate LLM response format and content"""
    if not response or len(response.strip()) < 10:
        return False, "Response too short or empty"
    
    if expected_format == 'json':
        try:
            json.loads(response)
            return True, "Valid JSON"
        except:
            return False, "Invalid JSON format"
    
    return True, "Response validated"

def extract_trading_signals_from_text(text):
    """Extract trading signals from LLM text response"""
    signals = []
    text_lower = text.lower()
    
    # Simple keyword extraction
    if any(word in text_lower for word in ['buy', 'long', 'bullish', 'accumulate']):
        signals.append('BUY')
    
    if any(word in text_lower for word in ['sell', 'short', 'bearish', 'reduce']):
        signals.append('SELL')
    
    if any(word in text_lower for word in ['hold', 'wait', 'neutral', 'sideways']):
        signals.append('HOLD')
    
    return signals if signals else ['HOLD']

def format_market_data_for_llm(market_data):
    """Format market data for LLM consumption"""
    formatted = {}
    
    for key, value in market_data.items():
        if isinstance(value, (int, float)):
            if abs(value) < 0.01:
                formatted[key] = f"{value:.4f}"
            elif abs(value) < 1:
                formatted[key] = f"{value:.3f}"
            else:
                formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = str(value)
    
    return formatted
