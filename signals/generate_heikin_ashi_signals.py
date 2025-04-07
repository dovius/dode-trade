import pandas as pd
import numpy as np

def generate_heikin_ashi_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on Heikin-Ashi candlestick patterns.
    
    Heikin-Ashi candlesticks are modified candlesticks that filter out market noise
    to better identify trends. This strategy generates buy signals when candles change
    from red to green (indicating a potential uptrend) and sell signals when candles
    change from green to red (indicating a potential downtrend).
    
    Args:
        df (pd.DataFrame): OHLCV data
        num_signals (int): Not used in this implementation, kept for compatibility
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Calculate Heikin-Ashi candles
    ha_data = calculate_heikin_ashi(data)
    
    # Generate signals based on color changes in Heikin-Ashi candles
    ha_data['candle_color'] = np.where(ha_data['ha_close'] >= ha_data['ha_open'], 'green', 'red')
    ha_data['prev_candle_color'] = ha_data['candle_color'].shift(1)
    
    # Buy signal: candle changes from red to green (trend reversal to uptrend)
    buy_condition = (ha_data['candle_color'] == 'green') & (ha_data['prev_candle_color'] == 'red')
    
    # Additional confirmation: current candle has no lower shadow (strong bullish)
    buy_condition = buy_condition & (ha_data['ha_low'] == ha_data['ha_open'])
    
    # Sell signal: candle changes from green to red (trend reversal to downtrend)
    sell_condition = (ha_data['candle_color'] == 'red') & (ha_data['prev_candle_color'] == 'green')
    
    # Additional confirmation: current candle has no upper shadow (strong bearish)
    sell_condition = sell_condition & (ha_data['ha_high'] == ha_data['ha_open'])
    
    # Extract buy and sell points
    buy_points = ha_data[buy_condition].copy()
    sell_points = ha_data[sell_condition].copy()
    
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

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Heikin-Ashi candles from regular OHLC data.
    
    Heikin-Ashi Formula:
    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (Previous HA Open + Previous HA Close) / 2
    - HA High = max(High, HA Open, HA Close)
    - HA Low = min(Low, HA Open, HA Close)
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        pd.DataFrame: DataFrame with original and Heikin-Ashi data
    """
    ha_data = df.copy()
    
    # Initialize Heikin-Ashi columns
    ha_data['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate HA Open
    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]  # First candle
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha_data['ha_close'].iloc[i-1]) / 2)
    ha_data['ha_open'] = ha_open
    
    # Calculate HA High and Low
    ha_data['ha_high'] = ha_data.apply(
        lambda x: max(x['high'], x['ha_open'], x['ha_close']), axis=1
    )
    ha_data['ha_low'] = ha_data.apply(
        lambda x: min(x['low'], x['ha_open'], x['ha_close']), axis=1
    )
    
    return ha_data
