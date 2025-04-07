import pandas as pd
import numpy as np

def generate_bollinger_bands_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on Bollinger Bands.
    
    This strategy generates buy signals when price touches the lower band and then
    moves back inside the bands, and sell signals when price touches the upper band
    and then moves back inside the bands.
    
    Args:
        df (pd.DataFrame): OHLCV data
        num_signals (int): Not used in this implementation, kept for compatibility
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Calculate Bollinger Bands
    period = 20
    std_dev = 2
    
    # Calculate the simple moving average
    data['sma'] = data['close'].rolling(window=period, min_periods=1).mean()
    
    # Calculate the standard deviation
    data['std'] = data['close'].rolling(window=period, min_periods=1).std()
    
    # Calculate the upper and lower bands
    data['upper_band'] = data['sma'] + (data['std'] * std_dev)
    data['lower_band'] = data['sma'] - (data['std'] * std_dev)
    
    # Identify when price is outside the bands
    data['outside_lower'] = data['close'] < data['lower_band']
    data['outside_upper'] = data['close'] > data['upper_band']
    
    # Identify when price moves back inside the bands
    data['prev_outside_lower'] = data['outside_lower'].shift(1)
    data['prev_outside_upper'] = data['outside_upper'].shift(1)
    
    # Buy signal: price was below lower band and now moved back inside
    buy_condition = (~data['outside_lower']) & (data['prev_outside_lower'])
    
    # Sell signal: price was above upper band and now moved back inside
    sell_condition = (~data['outside_upper']) & (data['prev_outside_upper'])
    
    # Extract buy and sell points
    buy_points = data[buy_condition].copy()
    sell_points = data[sell_condition].copy()
    
    # Format the output DataFrames
    if not buy_points.empty:
        buy_points = buy_points[['timestamp', 'close']].rename(columns={'close': 'price'})
    else:
        buy_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    if not sell_points.empty:
        sell_points = sell_points[['timestamp', 'close']].rename(columns={'close': 'price'})
    else:
        sell_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    return buy_points, sell_points
