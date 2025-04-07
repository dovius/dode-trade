import pandas as pd
import numpy as np
from datetime import time, datetime, timedelta

def generate_london_breakout_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on the London Breakout strategy.
    
    This strategy identifies the high and low prices during the pre-London session (typically 
    3:00-7:00 GMT), and then generates buy signals when price breaks above the pre-session high 
    and sell signals when price breaks below the pre-session low during the London session.
    
    Args:
        df (pd.DataFrame): OHLCV data with timestamp index
        num_signals (int): Not used in this implementation, kept for compatibility
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create UTC time objects for the London pre-session (3:00-7:00 GMT)
    pre_session_start = time(3, 0)  # 3:00 GMT
    pre_session_end = time(7, 0)    # 7:00 GMT
    
    # Create UTC time objects for the London session (7:00-16:00 GMT)
    session_start = time(7, 0)      # 7:00 GMT
    session_end = time(16, 0)       # 16:00 GMT
    
    # Initialize columns for signals
    data['pre_session'] = False
    data['london_session'] = False
    data['pre_session_high'] = np.nan
    data['pre_session_low'] = np.nan
    data['signal'] = 0
    
    # Group by date to process each day separately
    data['date'] = data['timestamp'].dt.date
    
    # Process each day
    buy_points_list = []
    sell_points_list = []
    
    for date, group in data.groupby('date'):
        # Convert date to datetime for comparison
        date_dt = datetime.combine(date, time(0, 0))
        
        # Identify pre-session candles
        pre_session_mask = group['timestamp'].apply(
            lambda x: pre_session_start <= x.time() < pre_session_end
        )
        
        # Skip if no pre-session data for this day
        if not pre_session_mask.any():
            continue
        
        # Calculate pre-session high and low
        pre_session_data = group[pre_session_mask]
        pre_session_high = pre_session_data['high'].max()
        pre_session_low = pre_session_data['low'].min()
        
        # Identify London session candles
        london_session_mask = group['timestamp'].apply(
            lambda x: session_start <= x.time() < session_end
        )
        
        # Skip if no London session data for this day
        if not london_session_mask.any():
            continue
        
        london_session_data = group[london_session_mask].copy()
        
        # Apply pre-session high and low to all London session candles
        london_session_data['pre_session_high'] = pre_session_high
        london_session_data['pre_session_low'] = pre_session_low
        
        # Generate signals
        # Buy when price breaks above pre-session high
        buy_breakout = london_session_data[
            (london_session_data['high'] > london_session_data['pre_session_high']) &
            (london_session_data['signal'] != 1)  # Avoid duplicate signals
        ].copy()
        
        if not buy_breakout.empty:
            # Take the first breakout of the day
            first_buy = buy_breakout.iloc[0]
            buy_point = pd.DataFrame({
                'timestamp': [first_buy['timestamp']],
                'price': [first_buy['pre_session_high']]  # Entry at breakout level
            })
            buy_points_list.append(buy_point)
        
        # Sell when price breaks below pre-session low
        sell_breakout = london_session_data[
            (london_session_data['low'] < london_session_data['pre_session_low']) &
            (london_session_data['signal'] != -1)  # Avoid duplicate signals
        ].copy()
        
        if not sell_breakout.empty:
            # Take the first breakout of the day
            first_sell = sell_breakout.iloc[0]
            sell_point = pd.DataFrame({
                'timestamp': [first_sell['timestamp']],
                'price': [first_sell['pre_session_low']]  # Entry at breakout level
            })
            sell_points_list.append(sell_point)
    
    # Combine all buy and sell points
    if buy_points_list:
        buy_points = pd.concat(buy_points_list, ignore_index=True)
    else:
        buy_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    if sell_points_list:
        sell_points = pd.concat(sell_points_list, ignore_index=True)
    else:
        sell_points = pd.DataFrame(columns=['timestamp', 'price'])
    
    return buy_points, sell_points