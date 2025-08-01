import requests
import json
import time
import os # Keep os for local testing, though st.secrets is preferred for cloud
import streamlit as st # Import streamlit to access st.secrets

# Gemini API configuration
# Fallback to os.getenv for local testing with a .env file
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="

def generate_market_commentary(recent_trades, current_market_conditions):
    """
    Generates market commentary using the Gemini API based on recent trading activity.
    """
    if not API_KEY:
        print("Warning: Gemini API Key is not set. Cannot generate market commentary.")
        return "Market commentary unavailable: API key missing or not set in environment/secrets."

    # Format recent_trades for the prompt
    formatted_trades = []
    for ts, trade_info in recent_trades:
        # Ensure ts is a Timestamp object before formatting
        if isinstance(ts, pd.Timestamp):
            formatted_trades.append(f"At {ts.strftime('%Y-%m-%d %H:%M')}, {trade_info['type']} {trade_info['qty']} shares at ${trade_info['price']:.2f}")
        else:
            # Handle cases where timestamp might be a string if loaded from JSON
            formatted_trades.append(f"At {ts}, {trade_info['type']} {trade_info['qty']} shares at ${trade_info['price']:.2f}")


    prompt = f"""
    Generate a concise, professional market commentary (1-2 sentences) for a high-frequency trading system.
    Focus on recent trading activity and its implications for the asset.

    Recent Trades: {"; ".join(formatted_trades) if formatted_trades else "No recent trades."}
    Current Market Conditions: Price: ${current_market_conditions['price']:.2f}, Prediction: {'UP' if current_market_conditions['prediction'] == 1 else 'DOWN/FLAT'}, Probability: {current_market_conditions['prediction_proba']:.2f}, Portfolio Value: ${current_market_conditions['portfolio_value']:.2f}.

    Example: "NVDA saw strong buying pressure in the last hour, indicating bullish sentiment. The system executed several long positions."
    """

    # The original JS fetch call structure needs to be adapted for Python requests library
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    headers = {
        'Content-Type': 'application/json'
    }

    # Implement exponential backoff for API calls
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = requests.post(API_URL + API_KEY, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
                commentary = result['candidates'][0]['content']['parts'][0]['text']
                return commentary
            else:
                print(f"LLM API response missing expected structure: {result}")
                return "Market commentary could not be generated due to unexpected API response."

        except requests.exceptions.RequestException as e:
            print(f"LLM API request failed: {e}")
            retries += 1
            sleep_time = 2 ** retries # Exponential backoff
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"An unexpected error occurred during LLM API call: {e}")
            return "Market commentary could not be generated due to an internal error."
            
    return "Market commentary failed after multiple retries."