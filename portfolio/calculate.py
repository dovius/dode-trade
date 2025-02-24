import pandas as pd

def calculate_portfolio_value(df: pd.DataFrame, buy_points: pd.DataFrame, sell_points: pd.DataFrame, initial_cash: float = 1000.0) -> pd.DataFrame:
    """
    Calculate portfolio value over time based on buy/sell signals.
    
    Args:
        df (pd.DataFrame): OHLCV data
        buy_points (pd.DataFrame): Buy signals
        sell_points (pd.DataFrame): Sell signals
        initial_cash (float): Starting cash amount
    
    Returns:
        pd.DataFrame: Portfolio value with cash, crypto amount, and total value
    """
    portfolio = pd.DataFrame(index=df.index)
    portfolio['cash'] = initial_cash
    portfolio['crypto_amount'] = 0.0
    portfolio['crypto_value'] = 0.0
    portfolio['total_value'] = initial_cash
    
    current_cash = initial_cash
    current_crypto = 0.0
    
    buy_timestamps = set(buy_points['timestamp'])
    sell_timestamps = set(sell_points['timestamp'])
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['close']
        
        if timestamp in buy_timestamps:
            amount_to_buy = current_cash * 0.95
            crypto_bought = amount_to_buy / price
            current_crypto += crypto_bought
            current_cash -= amount_to_buy
            print(f"Buy at {timestamp}: Bought {crypto_bought:.6f} units")
        
        elif timestamp in sell_timestamps:
            amount_received = current_crypto * price
            current_cash += amount_received
            current_crypto = 0.0
            print(f"Sell at {timestamp}: Sold for {amount_received:.2f}")
        
        portfolio.loc[idx, 'cash'] = current_cash
        portfolio.loc[idx, 'crypto_amount'] = current_crypto
        portfolio.loc[idx, 'crypto_value'] = current_crypto * price
        portfolio.loc[idx, 'total_value'] = current_cash + portfolio.loc[idx, 'crypto_value']
    
    return portfolio