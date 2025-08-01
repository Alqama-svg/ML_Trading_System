import pandas as pd

# Global portfolio state (for simplicity in this single-file simulation)
# In a real system, this would be stored in a database or dedicated state management service.
portfolio_state = {
    'cash': 100000.0,
    'shares_held': 0,
    'net_worth_history': [], # Will store tuples of (timestamp, net_worth)
    'trade_history': []      # Will store tuples of (timestamp, trade_info)
}

def reset_portfolio():
    portfolio_state['cash'] = 100000.0
    portfolio_state['shares_held'] = 0
    portfolio_state['net_worth_history'] = []
    portfolio_state['trade_history'] = []

def execute_trading_strategy(prediction, prediction_proba, current_price, timestamp):
    action = "HOLD"
    order_size = 0
    trade_info = {}

    BUY_THRESHOLD = 0.55
    SELL_THRESHOLD = 0.45

    if prediction == 1 and prediction_proba > BUY_THRESHOLD:
        action = "BUY"
        if portfolio_state['cash'] >= current_price * 10:
            order_size = 10
            trade_info = {'type': action, 'qty': order_size, 'price': current_price}
            # print(f"  ACTION: {action} {order_size} shares at ${current_price:.2f} (Prob: {prediction_proba:.2f})")
        else:
            action = "HOLD"
            # print(f"  ACTION: HOLD (Insufficient cash to buy 10 shares)")

    elif prediction == 0 and prediction_proba < SELL_THRESHOLD:
        action = "SELL"
        if portfolio_state['shares_held'] > 0:
            order_size = portfolio_state['shares_held']
            trade_info = {'type': action, 'qty': order_size, 'price': current_price}
            # print(f"  ACTION: {action} {order_size} shares at ${current_price:.2f} (Prob: {prediction_proba:.2f})")
        else:
            action = "HOLD"
            # print(f"  ACTION: HOLD (No shares to sell)")
    else:
        # print(f"  ACTION: HOLD (Prediction: {prediction}, Prob: {prediction_proba:.2f} - within neutral zone)")
        pass # No action, no print for hold

    # Update portfolio based on action
    if action == "BUY" and order_size > 0:
        cost = order_size * current_price
        portfolio_state['cash'] -= cost
        portfolio_state['shares_held'] += order_size
        portfolio_state['trade_history'].append((timestamp, trade_info))
    elif action == "SELL" and order_size > 0:
        revenue = order_size * current_price
        portfolio_state['cash'] += revenue
        portfolio_state['shares_held'] -= order_size
        portfolio_state['trade_history'].append((timestamp, trade_info))
    
    # Calculate current net worth
    current_net_worth = portfolio_state['cash'] + (portfolio_state['shares_held'] * current_price)
    portfolio_state['net_worth_history'].append((timestamp, current_net_worth))
    
    return action, order_size, current_net_worth

def get_current_portfolio_value(current_price):
    return portfolio_state['cash'] + (portfolio_state['shares_held'] * current_price)

def get_portfolio_history():
    return portfolio_state['net_worth_history'], portfolio_state['trade_history']
