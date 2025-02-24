import sqlite3
import pandas as pd

def save_to_sqlite(df: pd.DataFrame, symbol: str, timeframe: str, db_name: str = 'crypto_data.db') -> None:
    """
    Save DataFrame to SQLite database.
    
    Args:
        df (pd.DataFrame): Data to save
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '1m')
        db_name (str): Database file name
    """
    table_name = f"{symbol.replace('/', '_')}_{timeframe}"
    conn = sqlite3.connect(db_name)
    try:
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Data for {symbol} ({timeframe}) saved to {db_name}")
    except Exception as e:
        print(f"Error saving data to {db_name}: {e}")
    finally:
        conn.close()