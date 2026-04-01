import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

STOCKS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'WIPRO.NS']

def load_data(live=False):
    all_data = {}
    
    if live:
        # Live mode: last 6 months of real data
        end = datetime.today().strftime('%Y-%m-%d')
        start = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')
        print(f"🔴 LIVE MODE: Loading real NSE data from {start} to {end}")
    else:
        # Historical mode
        start = '2022-01-01'
        end = '2024-01-01'
        print(f"📊 HISTORICAL MODE: Loading data from {start} to {end}")

    for ticker in STOCKS:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['RSI'] = compute_rsi(df['Close'])
        df.dropna(inplace=True)
        all_data[ticker] = df
        print(f"✅ Loaded {ticker}: {len(df)} trading days")

    return all_data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def get_observation(all_data, step):
    obs = []
    for ticker in STOCKS:
        df = all_data[ticker]
        if step >= len(df):
            step = len(df) - 1
        row = df.iloc[step]
        close = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
        volume = float(row['Volume'].iloc[0]) if hasattr(row['Volume'], 'iloc') else float(row['Volume'])
        ma20 = float(row['MA20'].iloc[0]) if hasattr(row['MA20'], 'iloc') else float(row['MA20'])
        rsi = float(row['RSI'].iloc[0]) if hasattr(row['RSI'], 'iloc') else float(row['RSI'])
        obs.append([close, volume, ma20, rsi])
    return np.array(obs, dtype=np.float32)

def get_daily_returns(all_data, step):
    returns = []
    for ticker in STOCKS:
        df = all_data[ticker]
        if step < 1 or step >= len(df):
            returns.append(0.0)
        else:
            prev = df.iloc[step - 1]['Close']
            curr = df.iloc[step]['Close']
            prev = float(prev.iloc[0]) if hasattr(prev, 'iloc') else float(prev)
            curr = float(curr.iloc[0]) if hasattr(curr, 'iloc') else float(curr)
            returns.append((curr - prev) / (prev + 1e-9))
    return np.array(returns)

def get_latest_prices():
    """Get today's live prices for display"""
    prices = {}
    for ticker in STOCKS:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if len(hist) > 0:
                prices[ticker] = round(float(hist['Close'].iloc[-1]), 2)
            else:
                prices[ticker] = 'N/A'
        except:
            prices[ticker] = 'N/A'
    return prices