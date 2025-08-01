import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import time
import matplotlib.pyplot as plt # For potential matplotlib plots if needed

# Import modules from src/
from src.data_utils import mock_realtime_data_feed, initialize_data_buffer, data_buffer
from src.feature_calculator import compute_realtime_features
from src.model_inference import load_ml_models, get_ml_prediction
from src.strategy_executor import execute_trading_strategy, get_current_portfolio_value, get_portfolio_history, reset_portfolio
from src.llm_integration import generate_market_commentary

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', category=DeprecationWarning)

st.set_page_config(layout="wide", page_title="ML-Powered Trading System")

st.title("📈 ML-Powered Trading System")

# Global State for Simulation
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'processed_steps_count' not in st.session_state:
    st.session_state.processed_steps_count = 0
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = {
        'cash': 100000.0,
        'shares_held': 0,
        'net_worth_history': [],
        'trade_history': []
    }

SIMULATION_LIMIT = 500 # Max steps for a single simulation run

# Initialization Logic
@st.cache_resource # Cache resource to load models only once
def initialize_system_components():
    """Initializes the trading system components (models, data buffer)."""
    st.write(" Initializing Trading System Components ")
    
    # 1. Load ML Models
    models_loaded_status = load_ml_models()
    if not models_loaded_status:
        st.error("System initialization failed: ML models could not be loaded. Please ensure 'ML_Model_Implementation.ipynb' was run and saved models.")
        return False, None

    # 2. Initialize Data Buffer and Generator
    data_gen = initialize_data_buffer()
    st.write("System initialized successfully. Ready to simulate.")
    return True, data_gen

# Run initialization once
if not st.session_state.models_loaded:
    st.session_state.models_loaded, st.session_state.data_generator = initialize_system_components()
    if st.session_state.models_loaded:
        # Reset portfolio only if models loaded successfully for the very first time
        reset_portfolio() # Reset the global portfolio state in strategy_executor
        st.session_state.portfolio_data['net_worth_history'].append((pd.Timestamp.now(), 100000.0)) # Initial net worth for chart

# Functions for Simulation Steps
def process_single_tick():
    """
    Processes a single data tick in the simulation.
    """
    if not st.session_state.simulation_running:
        st.warning("Simulation not running. Please start it first.")
        return

    if st.session_state.processed_steps_count >= SIMULATION_LIMIT:
        st.session_state.simulation_running = False
        st.success("Simulation reached its limit. Complete.")
        return

    if st.session_state.data_generator is None:
        st.error("Data generator not initialized.")
        st.session_state.simulation_running = False
        return

    try:
        timestamp, new_row = next(st.session_state.data_generator)
    except StopIteration:
        st.session_state.simulation_running = False
        st.success("Data feed exhausted. Simulation complete.")
        return
    except Exception as e:
        st.error(f"Error fetching next data point: {e}")
        st.session_state.simulation_running = False
        return

    current_price = new_row['Close']
    st.session_state.processed_steps_count += 1

    # 1. Real-time Feature Computation
    latest_features_df = compute_realtime_features(new_row.copy(), data_buffer)

    if latest_features_df is None:
        st.info(f"[{timestamp}] Skipping prediction: Features not ready (e.g., NaNs or insufficient data).")
        # Update net worth history for display continuity
        st.session_state.portfolio_data['net_worth_history'].append((timestamp, get_current_portfolio_value(current_price)))
        return

    # 2. ML Model Inference
    prediction, prediction_proba = get_ml_prediction(latest_features_df)

    if prediction is None:
        st.info(f"[{timestamp}] Skipping strategy: Prediction failed.")
        st.session_state.portfolio_data['net_worth_history'].append((timestamp, get_current_portfolio_value(current_price)))
        return

    # 3. Strategy Execution (updates global portfolio_state from strategy_executor.py)
    action, order_size, current_net_worth = execute_trading_strategy(prediction, prediction_proba, current_price, timestamp)

    # Update Streamlit's session state with the latest portfolio data
    # Note: get_portfolio_history() returns (net_worth_history, trade_history)
    # net_worth_history is a list of (timestamp, value) tuples
    # trade_history is a list of (timestamp, trade_info_dict) tuples
    
    # Update cash and shares held from the global portfolio_state in strategy_executor
    st.session_state.portfolio_data['cash'] = get_portfolio_history()[0][-1][1] if get_portfolio_history()[0] else 100000.0
    st.session_state.portfolio_data['shares_held'] = get_portfolio_history()[1][-1][1]['qty'] if get_portfolio_history()[1] and get_portfolio_history()[1][-1][1]['type'] == 'BUY' else get_portfolio_history()[0][-1][1] if get_portfolio_history()[0] else 0.0
    st.session_state.portfolio_data['net_worth_history'] = get_portfolio_history()[0]
    st.session_state.portfolio_data['trade_history'] = get_portfolio_history()[1]


    # Generate market commentary (LLM Integration)
    recent_trades_for_llm = st.session_state.portfolio_data['trade_history'][-5:] # Last 5 trades
    current_market_conditions = {
        "price": current_price,
        "prediction": prediction,
        "prediction_proba": prediction_proba,
        "portfolio_value": current_net_worth
    }
    market_commentary = generate_market_commentary(recent_trades_for_llm, current_market_conditions)

    # Update UI elements
    st.session_state.current_timestamp = str(timestamp)
    st.session_state.current_price = f"${current_price:.2f}"
    st.session_state.ml_prediction = f"{prediction} ({'UP' if prediction == 1 else 'DOWN/FLAT'})"
    st.session_state.prediction_proba = f"{prediction_proba:.2f}"
    st.session_state.action_executed = action
    st.session_state.portfolio_value = f"${current_net_worth:.2f}"
    st.session_state.market_commentary = market_commentary
    st.session_state.cash = f"${st.session_state.portfolio_data['cash']:.2f}"
    st.session_state.shares_held = st.session_state.portfolio_data['shares_held']

# --- Control Panel ---
st.sidebar.header("Control Panel")

if st.sidebar.button("Start Simulation", disabled=st.session_state.simulation_running or not st.session_state.models_loaded):
    if st.session_state.models_loaded:
        st.session_state.simulation_running = True
        st.session_state.processed_steps_count = 0
        reset_portfolio() # Reset global portfolio state
        st.session_state.portfolio_data['net_worth_history'] = [(pd.Timestamp.now(), 100000.0)] # Initial net worth for chart
        st.session_state.portfolio_data['trade_history'] = [] # Clear trade history
        st.session_state.data_generator = initialize_data_buffer() # Re-initialize data generator
        st.sidebar.success("Simulation started!")
    else:
        st.sidebar.error("Models not loaded. Cannot start simulation.")

if st.sidebar.button("Stop Simulation", disabled=not st.session_state.simulation_running):
    st.session_state.simulation_running = False
    st.sidebar.info("Simulation stopped.")

if st.sidebar.button("Process Next Tick (Manual)", disabled=st.session_state.simulation_running or not st.session_state.models_loaded):
    process_single_tick()


# --- Real-time Metrics Display ---
st.header("Real-time Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current Timestamp", st.session_state.get('current_timestamp', 'N/A'))
    st.metric("ML Prediction", st.session_state.get('ml_prediction', 'N/A'))
    st.metric("Action Executed", st.session_state.get('action_executed', 'N/A'))
with col2:
    st.metric("Current Price", st.session_state.get('current_price', 'N/A'))
    st.metric("Prediction Probability", st.session_state.get('prediction_proba', 'N/A'))
    st.metric("Portfolio Value", st.session_state.get('portfolio_value', 'N/A'))
with col3:
    st.metric("Cash", st.session_state.get('cash', 'N/A'))
    st.metric("Shares Held", st.session_state.get('shares_held', 'N/A'))

# --- Market Commentary ---
st.header("Market Commentary (LLM)")
st.info(st.session_state.get('market_commentary', 'Awaiting market data for commentary...'))

# --- Portfolio Performance Chart ---
st.header("Portfolio Performance")
net_worth_df_chart = pd.DataFrame(st.session_state.portfolio_data['net_worth_history'], columns=['Timestamp', 'Value'])
if not net_worth_df_chart.empty:
    net_worth_df_chart.set_index('Timestamp', inplace=True)
    st.line_chart(net_worth_df_chart['Value'])
else:
    st.write("No portfolio data to display yet.")

# --- Trade History ---
st.header("Trade History")
trade_history_display = []
for ts, info in st.session_state.portfolio_data['trade_history']:
    trade_history_display.append({
        'Timestamp': str(ts),
        'Action': info['type'],
        'Quantity': info['qty'],
        'Price': f"${info['price']:.2f}"
    })
if trade_history_display:
    st.dataframe(pd.DataFrame(trade_history_display).set_index('Timestamp').sort_index(ascending=False))
else:
    st.write("No trades executed yet.")

# --- Auto-refresh logic for continuous simulation ---
if st.session_state.simulation_running:
    # This reruns the script every 500ms to simulate real-time updates
    time.sleep(0.5)
    st.rerun() # Use st.rerun() for immediate re-execution