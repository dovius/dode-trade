import pandas as pd
import numpy as np

def generate_rsi_signals(df: pd.DataFrame, num_signals: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate trading signals based on Relative Strength Index (RSI) indicator.
    
    This strategy generates buy signals when RSI crosses above the oversold threshold (30)
    and sell signals when RSI crosses below the overbought threshold (70).
    
    Args:
        df (pd.DataFrame): OHLCV data
        num_signals (int): Not used in this implementation, kept for compatibility
    
    Returns:
        tuple: (buy_points, sell_points) as DataFrames
    """
    # Create a copy of the DataFrame to avoid modifying the original
    data = df.copy()
    
    # Calculate RSI with a period of 14
    period = 14
    data['price_change'] = data['close'].diff()
    data['gain'] = data['price_change'].apply(lambda x: x if x > 0 else 0)
    data['loss'] = data['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate average gain and average loss
    data['avg_gain'] = data['gain'].rolling(window=period, min_periods=1).mean()
    data['avg_loss'] = data['loss'].rolling(window=period, min_periods=1).mean()
    
    # Calculate relative strength (RS) and RSI
    data['rs'] = data['avg_gain'] / data['avg_loss'].replace(0, 0.001)  # Avoid division by zero
    data['rsi'] = 100 - (100 / (1 + data['rs']))
    
    # Define overbought and oversold thresholds
    overbought = 70
    oversold = 30
    
    # Generate signals
    data['rsi_signal'] = 0
    data.loc[data['rsi'] < oversold, 'rsi_signal'] = 1  # Oversold (potential buy)
    data.loc[data['rsi'] > overbought, 'rsi_signal'] = -1  # Overbought (potential sell)
    
    # Find crossovers (signal changes)
    data['prev_rsi_signal'] = data['rsi_signal'].shift(1)
    
    # Buy when RSI crosses above oversold threshold (30)
    buy_condition = (data['rsi_signal'] == 0) & (data['prev_rsi_signal'] == 1)
    
    # Sell when RSI crosses below overbought threshold (70)
    sell_condition = (data['rsi_signal'] == 0) & (data['prev_rsi_signal'] == -1)
    
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
