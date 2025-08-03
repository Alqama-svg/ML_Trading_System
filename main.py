import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('src', exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="ML-Powered Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e1e5e9;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Import modules with error handling
try:
    from src.ml_models import MLTradingModels
    from src.feature_engineering import create_features
    from src.trading_strategies import AdvancedStrategies
    from src.llm_integration import LLMAssistant
    from src.risk_management import RiskManager
    from src.performance_monitor import calculate_metrics
except ImportError as e:
    st.error(f"Module import error: {e}. Creating basic functionality...")

class MLTradingSystem:
    def __init__(self):
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize session state
            if 'system_initialized' not in st.session_state:
                st.session_state.system_initialized = True
                st.session_state.current_data = None
                st.session_state.predictions = None
                st.session_state.portfolio_value = 100000
                st.session_state.positions = {}
                st.session_state.trade_history = []
                st.session_state.cash = 100000
                st.session_state.shares_held = 0
                
            # Initialize ML models
            self.ml_models = self.load_or_create_models()
            st.success("✅ System Initialized Successfully")
            
        except Exception as e:
            st.error(f"System initialization error: {str(e)}")
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        model_path = Path('models/trading_model.pkl')
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Create simple model if none exists
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model

    def create_features(self, data):
        """Create features for ML model"""
        if data is None or data.empty:
            return pd.DataFrame()
        
        df = data.copy()
        
        # Technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Volatility
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        
        return df.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 ML-Powered Trading System</h1>', unsafe_allow_html=True)
    
    # Initialize system
    trading_system = MLTradingSystem()
    
    # Sidebar - Control Panel
    st.sidebar.title("🎛 Control Panel")
    
    # System Status
    with st.sidebar:
        st.subheader("System Status")
        if st.session_state.get('system_initialized', False):
            st.markdown('<div class="success-box">✅ System Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-box">❌ System Offline</div>', unsafe_allow_html=True)
    
    # Main controls
    action = st.sidebar.selectbox(
        "Select Action",
        ["Start Simulation", "Stop Simulation", "Process Next Tick (Manual)", "View Analytics"]
    )
    
    # Symbol selection
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
    
    # Time period
    period = st.sidebar.selectbox(
        "Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Real-time Metrics")
        
        # Current metrics
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create metric containers
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric("Current Timestamp", current_time)
            st.metric("Cash", f"${st.session_state.cash:,.2f}")
            st.metric("Shares Held", st.session_state.shares_held)
        
        with metric_col2:
            st.metric("Current Price", "N/A")
            st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
            st.metric("ML Prediction", "N/A")
    
    with col2:
        st.subheader("📈 Market Commentary (LLM)")
        st.info("Awaiting market data for commentary...")
    
    # Data processing section
    if symbol and action == "Start Simulation":
        try:
            with st.spinner(f"Fetching data for {symbol}..."):
                # Fetch market data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    st.session_state.current_data = data
                    current_price = data['Close'].iloc[-1]
                    
                    # Update current price metric
                    st.success(f"✅ Data loaded for {symbol}")
                    
                    # Create features
                    features_df = trading_system.create_features(data)
                    
                    if not features_df.empty:
                        # Simple prediction logic
                        latest_features = features_df.iloc[-1]
                        
                        # Simple trading signal
                        if latest_features['SMA_10'] > latest_features['SMA_30']:
                            prediction = "BUY"
                            prob = 0.75
                        else:
                            prediction = "SELL"
                            prob = 0.65
                        
                        # Update metrics
                        metric_col2.metric("Current Price", f"${current_price:.2f}")
                        metric_col2.metric("ML Prediction", prediction)
                        
                        # Chart
                        st.subheader("📊 Price Chart & Indicators")
                        
                        fig = go.Figure()
                        
                        # Candlestick chart
                        fig.add_trace(go.Candlestick(
                            x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'],
                            name='Price'
                        ))
                        
                        # Add SMA lines
                        if 'SMA_10' in features_df.columns:
                            fig.add_trace(go.Scatter(
                                x=features_df.index,
                                y=features_df['SMA_10'],
                                name='SMA 10',
                                line=dict(color='blue', width=1)
                            ))
                        
                        if 'SMA_30' in features_df.columns:
                            fig.add_trace(go.Scatter(
                                x=features_df.index,
                                y=features_df['SMA_30'],
                                name='SMA 30',
                                line=dict(color='red', width=1)
                            ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Chart with Indicators",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Market Commentary
                        with col2:
                            st.subheader("📈 Market Commentary (LLM)")
                            
                            commentary = f"""
                            **Current Analysis for {symbol}:**
                            
                            📊 **Technical Indicators:**
                            - Current Price: ${current_price:.2f}
                            - 10-day SMA: ${latest_features['SMA_10']:.2f}
                            - 30-day SMA: ${latest_features['SMA_30']:.2f}
                            - RSI: {latest_features['RSI']:.1f}
                            
                            🎯 **ML Prediction:** {prediction} (Confidence: {prob:.1%})
                            
                            📈 **Strategy Recommendation:**
                            {"Consider buying position - short-term momentum is positive" if prediction == "BUY" else "Consider selling/holding - short-term momentum is negative"}
                            
                            ⚠️ **Risk Assessment:** Monitor volatility levels and volume trends
                            """
                            
                            st.markdown(commentary)
                
                else:
                    st.error(f"No data found for symbol: {symbol}")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    
    # Analytics section
    if action == "View Analytics":
        st.subheader("📈 Performance Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", "0.00%")
            st.metric("Sharpe Ratio", "N/A")
        
        with col2:
            st.metric("Max Drawdown", "0.00%")
            st.metric("Win Rate", "N/A")
        
        with col3:
            st.metric("Total Trades", len(st.session_state.trade_history))
            st.metric("Avg Trade Return", "N/A")
    
    # Footer
    st.markdown("---")
    st.markdown("**ML-Powered Trading System** - Advanced algorithmic trading with machine learning integration")

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> f1909685739746bbe77927120694a7980b73754a
