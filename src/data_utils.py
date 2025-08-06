import time # Import time module

def mock_realtime_data_feed(ticker="NVDA", period="7d", interval="1m"):
   
    print(f"Fetching historical data for {ticker} to simulate real-time feed...")
    df_hist = yf.Ticker(ticker).history(period=period, interval=interval)
    df_hist.dropna(inplace=True)
    print(f"Loaded {len(df_hist)} historical data points for simulation.")

    for index, row in df_hist.iterrows():
        yield index, row
        time.sleep(0.1) # Added a small delay, e.g., 100 milliseconds per tick. Adjust as needed.
