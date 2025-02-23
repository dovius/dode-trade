import sqlite3
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_from_sqlite(symbol, timeframe, db_name='crypto_data.db'):
    conn = sqlite3.connect(db_name)
    table_name = f"{symbol.replace('/', '_')}_{timeframe}"
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def generate_dummy_signals(df, num_signals=10):
    # Select random indices from df
    random_indices = random.sample(list(df.index), min(num_signals, len(df)))
    random_indices.sort()
    
    # Split into buy and sell points
    buy_indices = random_indices[0::2]
    sell_indices = random_indices[1::2]
    
    # Create buy and sell DataFrames
    buy_points = df.loc[buy_indices, ['timestamp', 'close']].rename(columns={'close': 'price'})
    sell_points = df.loc[sell_indices, ['timestamp', 'close']].rename(columns={'close': 'price'})
    
    # Debugging: Verify the signals
    print("Buy Points:\n", buy_points)
    print("Sell Points:\n", sell_points)
    
    return buy_points, sell_points

def calculate_portfolio_value(df, buy_points, sell_points, initial_cash=1000):
    # Initialize portfolio DataFrame with the same index as df
    portfolio = pd.DataFrame(index=df.index)
    portfolio['cash'] = initial_cash
    portfolio['crypto_amount'] = 0
    portfolio['crypto_value'] = 0
    portfolio['total_value'] = initial_cash
    
    current_cash = initial_cash
    current_crypto = 0
    
    # Create sets of timestamps for efficient lookup
    buy_timestamps = set(buy_points['timestamp'])
    sell_timestamps = set(sell_points['timestamp'])
    
    # Iterate over the DataFrame
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['close']
        
        # Buy transaction
        if timestamp in buy_timestamps:
            amount_to_buy = current_cash * 0.95  # Use 95% of cash
            crypto_bought = amount_to_buy / price
            current_crypto += crypto_bought
            current_cash -= amount_to_buy
            print(f"Buy at {timestamp}: Bought {crypto_bought} units")
        
        # Sell transaction
        elif timestamp in sell_timestamps:
            amount_received = current_crypto * price
            current_cash += amount_received
            current_crypto = 0
            print(f"Sell at {timestamp}: Sold for {amount_received}")
        
        # Update portfolio
        portfolio.loc[idx, 'cash'] = current_cash
        portfolio.loc[idx, 'crypto_amount'] = current_crypto
        portfolio.loc[idx, 'crypto_value'] = current_crypto * price
        portfolio.loc[idx, 'total_value'] = current_cash + portfolio.loc[idx, 'crypto_value']
    
    # Debugging: Check if total_value varies
    print("Portfolio Value Stats:\n", portfolio['total_value'].describe())
    
    return portfolio

if __name__ == "__main__":
    symbol = 'BTC/USDT'
    df = load_from_sqlite(symbol, "1m")
    initial_cash = 1000

    # Generate signals and calculate portfolio
    buy_points, sell_points = generate_dummy_signals(df, num_signals=10)
    portfolio = calculate_portfolio_value(df, buy_points, sell_points, initial_cash=initial_cash)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"{symbol} Price Chart with Signals", "Portfolio Value"),
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0,0,0,0.3)'
        ),
        row=1, col=1,
        secondary_y=True
    )

    # Add buy points
    if len(buy_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_points['timestamp'],
                y=buy_points['price'],
                mode='markers',
                name='Buy',
                marker=dict(symbol='triangle-up', size=15, color='green')
            ),
            row=1, col=1
        )

    # Add sell points
    if len(sell_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_points['timestamp'],
                y=sell_points['price'],
                mode='markers',
                name='Sell',
                marker=dict(symbol='triangle-down', size=15, color='red')
            ),
            row=1, col=1
        )

    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=portfolio['total_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )

    # Update layout and axes
    price_min = df['low'].min()
    price_max = df['high'].max()
    fig.update_layout(
        title_text="Crypto Trading Analysis",
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", range=[price_min, price_max], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)

    fig.show()