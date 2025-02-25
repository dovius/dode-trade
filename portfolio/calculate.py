import pandas as pd
import numpy as np

def calculate_portfolio_value(df: pd.DataFrame, buy_points: pd.DataFrame, sell_points: pd.DataFrame, initial_cash: float = 1000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate portfolio value over time based on buy/sell signals.
    Also tracks individual trade performance.
    
    Args:
        df (pd.DataFrame): OHLCV data
        buy_points (pd.DataFrame): Buy signals
        sell_points (pd.DataFrame): Sell signals
        initial_cash (float): Starting cash amount
    
    Returns:
        tuple: (portfolio_df, trades_df)
            - portfolio_df: Portfolio value with cash, crypto amount, and total value
            - trades_df: Individual trade performance data
    """
    portfolio = pd.DataFrame(index=df.index)
    portfolio['cash'] = initial_cash
    portfolio['crypto_amount'] = 0.0
    portfolio['crypto_value'] = 0.0
    portfolio['total_value'] = initial_cash
    
    # For tracking trades
    trades = []
    current_buy_price = None
    current_buy_time = None
    
    current_cash = initial_cash
    current_crypto = 0.0
    
    buy_timestamps = set(buy_points['timestamp'])
    sell_timestamps = set(sell_points['timestamp'])
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['close']
        
        if timestamp in buy_timestamps:
            amount_to_buy = current_cash * 1.0  # Using all cash
            crypto_bought = amount_to_buy / price
            fees = crypto_bought * 0.001  # 0.1% fee
            current_crypto += crypto_bought - fees
            current_cash -= amount_to_buy
            
            # Track buy
            current_buy_price = price
            current_buy_time = timestamp
            print(f"Buy at {timestamp}: Bought {crypto_bought:.6f} units")
        
        elif timestamp in sell_timestamps and current_crypto > 0:
            amount_received = current_crypto * price
            fees = amount_received * 0.001  # 0.1% fee
            current_cash += amount_received - fees
            
            # Calculate trade productivity
            if current_buy_price is not None:
                buy_value = current_crypto * current_buy_price
                sell_value = amount_received
                profit_loss = sell_value - buy_value
                profit_percentage = (price / current_buy_price - 1) * 100
                
                # Record trade
                trades.append({
                    'buy_time': current_buy_time,
                    'sell_time': timestamp,
                    'buy_price': current_buy_price,
                    'sell_price': price,
                    'amount': current_crypto,
                    'profit_loss': profit_loss,
                    'profit_percentage': profit_percentage,
                    'fees': fees
                })
            
            print(f"Sell at {timestamp}: Sold for {amount_received:.2f}")
            current_crypto = 0.0
            current_buy_price = None
            current_buy_time = None
        
        portfolio.loc[idx, 'cash'] = current_cash
        portfolio.loc[idx, 'crypto_amount'] = current_crypto
        portfolio.loc[idx, 'crypto_value'] = current_crypto * price
        portfolio.loc[idx, 'total_value'] = current_cash + portfolio.loc[idx, 'crypto_value']
    
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=['buy_time', 'sell_time', 'buy_price', 'sell_price', 
                 'amount', 'profit_loss', 'profit_percentage', 'fees'])
    
    return portfolio, trades_df