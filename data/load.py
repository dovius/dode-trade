import sqlite3
import pandas as pd

def load_from_sqlite(symbol: str, timeframe: str, db_name: str = 'crypto_data.db') -> pd.DataFrame:
    """
    Load data from SQLite database.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '1m')
        db_name (str): Database file name
    
    Returns:
        pd.DataFrame: Loaded OHLCV data
    """
    conn = sqlite3.connect(db_name)
    table_name = f"{symbol.replace('/', '_')}_{timeframe}"
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    except Exception as e:
        print(f"Error loading data from {db_name}: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df