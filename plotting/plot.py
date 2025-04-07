import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_data(df: pd.DataFrame, buy_points: pd.DataFrame, sell_points: pd.DataFrame, 
              portfolio: pd.DataFrame, symbol: str, trades_df: pd.DataFrame = None) -> None:
    """
    Plot price chart with signals, portfolio value, and trade profitability.
    
    Args:
        df (pd.DataFrame): OHLCV data
        buy_points (pd.DataFrame): Buy signals
        sell_points (pd.DataFrame): Sell signals
        portfolio (pd.DataFrame): Portfolio value data
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        trades_df (pd.DataFrame): Trade performance data
    """
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=(
            f"{symbol} Price Chart with Signals", 
            "Portfolio Value", 
            "Trade Profitability"
        ),
        row_heights=[0.5, 0.25, 0.25], 
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Price chart with candlesticks and volume
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], 
            low=df['low'], close=df['close'], name='Price'
        ), 
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'], y=df['volume'], name='Volume', 
            marker_color='rgba(0,0,0,0.3)'
        ), 
        row=1, col=1, secondary_y=True
    )
    
    # Add signals as vertical lines
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
    
    # Portfolio value chart
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=portfolio['cash'], 
            name='Cash', line=dict(color='green', width=2)
        ), 
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=portfolio['crypto_value'], 
            name='Crypto Value', line=dict(color='orange', width=2)
        ), 
        row=2, col=1
    )
    
    # fig.add_trace(
    #     go.Scatter(
    #         x=df['timestamp'], y=portfolio['total_value'], 
    #         name='Total Value', line=dict(color='blue', width=2)
    #     ), 
    #     row=2, col=1
    # )
    
    # Pridedame tekstą, kuris rodo paskutinę vertę portfolyje
    last_timestamp = df['timestamp'].iloc[-1]
    last_cash = portfolio['cash'].iloc[-1]
    last_crypto = portfolio['crypto_value'].iloc[-1]
    
    # Pridedame anotacijas rodančias paskutines vertes
    fig.add_annotation(
        x=last_timestamp,
        y=last_cash,
        text=f"${last_cash:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        ax=40,
        ay=-40,
        row=2, col=1
    )
    
    fig.add_annotation(
        x=last_timestamp,
        y=last_crypto,
        text=f"${last_crypto:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="orange",
        ax=20,
        ay=-20,
        row=2, col=1
    )
    
    # Trade profitability chart
    if trades_df is not None and not trades_df.empty:
        # Create color scale based on profitability
        colors = ['red' if p < 0 else 'green' for p in trades_df['profit_percentage']]
        
        # Trade profit percentage bars
        fig.add_trace(
            go.Bar(
                x=trades_df['sell_time'],
                y=trades_df['profit_percentage'],
                name='Profit %',
                marker_color=colors,
                text=[f"{p:.2f}%" for p in trades_df['profit_percentage']],
                textposition='auto'
            ),
            row=3, col=1
        )
        
        # Add annotations for trade details
        for i, trade in trades_df.iterrows():
            fig.add_annotation(
                x=trade['sell_time'],
                y=trade['profit_percentage'],
                text=f"${trade['profit_loss']:.2f}",
                showarrow=False,
                yshift=10 if trade['profit_percentage'] >= 0 else -10,
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Crypto Trading Analysis",
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price", range=[price_min, price_max], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_yaxes(title_text="Profit/Loss (%)", row=3, col=1)
    
    fig.show()