import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_data(df: pd.DataFrame, buy_points: pd.DataFrame, sell_points: pd.DataFrame, portfolio: pd.DataFrame, symbol: str) -> None:
    """
    Plot price chart with signals and portfolio value.
    
    Args:
        df (pd.DataFrame): OHLCV data
        buy_points (pd.DataFrame): Buy signals
        sell_points (pd.DataFrame): Sell signals
        portfolio (pd.DataFrame): Portfolio value data
        symbol (str): Trading pair (e.g., 'BTC/USDT')
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        subplot_titles=(f"{symbol} Price Chart with Signals", "Portfolio Value"),
        row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='rgba(0,0,0,0.3)'), row=1, col=1, secondary_y=True)
    
    # Assuming 'fig' is already defined (e.g., with price data plotted)
    price_min = df['low'].min()
    price_max = df['high'].max()

    shapes = []

    if not buy_points.empty:
        for ts in buy_points['timestamp']:
            shapes.append(dict(
                type='line',
                x0=ts,
                y0=price_min,
                x1=ts,
                y1=price_max,
                line=dict(color='green', width=2, dash='dash'),
                xref='x1',
                yref='y1'
            ))

    if not sell_points.empty:
        for ts in sell_points['timestamp']:
            shapes.append(dict(
                type='line',
                x0=ts,
                y0=price_min,
                x1=ts,
                y1=price_max,
                line=dict(color='red', width=2, dash='dot'),
                xref='x1',
                yref='y1'
            ))

    fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=portfolio['cash'], name='Cash', line=dict(color='green', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=portfolio['crypto_value'], name='Crypto Value', line=dict(color='orange', width=2)), row=2, col=1)    

    price_min, price_max = df['low'].min(), df['high'].max()
    fig.update_layout(title_text="Crypto Trading Analysis", height=1000, showlegend=True, xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", range=[price_min, price_max], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    
    fig.show()