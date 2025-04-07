import pandas as pd
import numpy as np

def generate_moving_avg_crossover_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on moving average crossover strategy.
    
    This strategy generates buy signals when the short-term moving average (9-period)
    crosses above the long-term moving average (21-period), and sell signals when the
    short-term moving average crosses below the long-term moving average.
    
    Args:
        df (pd.DataFrame): OHLCV data
        num_signals (int): Not used in this implementation, kept for compatibility
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Calculate moving averages (short 9-period, long 21-period)
    short_window = 450
    long_window = 1200
    data['short_ma'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data['long_ma'] = data['close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create signals
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    
    # Create buy and sell points on crossover
    data['position_change'] = data['signal'].diff()
    
    # Buy when short MA crosses above long MA
    buy_points = data[data['position_change'] == 1].copy()
    
    # Sell when short MA crosses below long MA
    sell_points = data[data['position_change'] == -1].copy()
    
    if not buy_points.empty:
        buy_points = buy_points[['timestamp', 'close']].rename(columns={'close': 'price'})
    else:
        buy_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    if not sell_points.empty:
        sell_points = sell_points[['timestamp', 'close']].rename(columns={'close': 'price'})
    else:
        sell_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    return buy_points, sell_points