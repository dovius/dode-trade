import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import argparse

import sqlite3
from datetime import datetime, timedelta

def save_to_sqlite(df, symbol, timeframe, db_name='crypto_data.db'):
    # Replace '/' with '_' in symbol name for SQL compatibility
    table_name = f"{symbol.replace('/', '_')}_{timeframe}"
    
    # Create connection to SQLite database
    conn = sqlite3.connect(db_name)
    
    try:
        # Save DataFrame to SQL
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists='replace',  # 'replace' will overwrite existing table, use 'append' to add data
            index=False
        )
        print(f"Data for {symbol} successfully saved to {db_name}")
    except Exception as e:
        print(f"Error saving data: {e}")
    finally:
        conn.close()


def fetch_historical_data(symbol, timeframe='1d', start_date=None, end_date=None):
    binance = ccxt.binance()
    
    start_timestamp = int(start_date.timestamp() * 1000) if start_date else None
    end_timestamp = int(end_date.timestamp() * 1000) if end_date else int(time.time() * 1000)
    
    all_candles = []
    current_timestamp = start_timestamp
    
    while current_timestamp < end_timestamp:
        try:
            candles = binance.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=current_timestamp,
                limit=1000
            )
            
            if not candles or len(candles) <= 1:
                break
                
            all_candles.extend(candles)
            current_timestamp = candles[-1][0]
            
            time.sleep(binance.rateLimit / 1000)

        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def analyze_historical_data(symbol, timeframe='1d', start_date=None, end_date=None):
    
    print(f"Fetching data for {symbol}...")
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    
    print(f"\nStatistics for {symbol}:")
    print(f"Maximum Price: ${df['high'].max():,.2f}")
    print(f"Minimum Price: ${df['low'].min():,.2f}")
    print("-" * 50)
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=7)
    parser.add_argument('--symbol', type=str, default='BTC')
    parser.add_argument('--timeframe', type=str, default='1m')
    args = parser.parse_args()

    symbol = f"{args.symbol}/USDT"
    start_date = datetime.now() - timedelta(days=args.days)
    end_date = datetime.now()
    timeframe = args.timeframe
    
    # Fetch, analyze, and visualize data
    historical_data = analyze_historical_data(
        symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    print(historical_data)

    save_to_sqlite(historical_data, symbol, timeframe)