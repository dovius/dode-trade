import pandas as pd
import random

def generate_dummy_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate random buy and sell signals.
    
    Args:
        df (pd.DataFrame): OHLCV data
        num_signals (int): Number of signals to generate
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    random_indices = random.sample(list(df.index), min(num_signals, len(df)))
    random_indices.sort()
    
    buy_indices = random_indices[0::2]
    sell_indices = random_indices[1::2]
    
    buy_points = df.loc[buy_indices, ['timestamp', 'close']].rename(columns={'close': 'price'})
    sell_points = df.loc[sell_indices, ['timestamp', 'close']].rename(columns={'close': 'price'})
    
    return buy_points, sell_points