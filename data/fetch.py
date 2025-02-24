import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_historical_data(symbol: str, timeframe: str = '1d', start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '1m', '1d')
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
    
    Returns:
        pd.DataFrame: OHLCV data with timestamp in datetime format
    """
    exchange = ccxt.binance()
    start_timestamp = int(start_date.timestamp() * 1000) if start_date else None
    end_timestamp = int(end_date.timestamp() * 1000) if end_date else int(time.time() * 1000)
    
    all_candles = []
    current_timestamp = start_timestamp
    
    while current_timestamp < end_timestamp:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=1000)
            if not candles or len(candles) <= 1:
                break
            all_candles.extend(candles)
            current_timestamp = candles[-1][0]
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            break
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df