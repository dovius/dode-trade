import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import plotly.io as pio

# Set the default renderer to 'browser'
pio.renderers.default = 'browser'

def plot_summary(fig: go.Figure | None, buy_points: pd.DataFrame, sell_points: pd.DataFrame, 
              portfolio: pd.DataFrame, symbol: str, alg_name: str) -> None:
    """
    Plot one dimensional buy/sell points chart and write resulting portfolio value next to that chart.
    
    Args:
        buy_points (pd.DataFrame): Buy signals
        sell_points (pd.DataFrame): Sell signals
        portfolio (pd.DataFrame): Portfolio value data
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        trades_df (pd.DataFrame): Trade performance data (optional)
    """
    percentage_change = (portfolio.iloc[-1]["total_value"] - portfolio.iloc[0]["total_value"]) / portfolio.iloc[0]["total_value"]

    if fig is None:
        fig = go.Figure()

    fig.update_layout(
        title=f'{symbol} - {alg_name} Signals. Change: {percentage_change:.2%}', 
        # Set a very small height for the chart
        height=100,  # Very small height in pixels
        margin=dict(l=20, r=20, t=50, b=10)  # Reduce margins to make chart more compact
    )
    
    if not buy_points.empty:
        fig.add_trace(go.Scatter(x=buy_points['timestamp'], y=[0] * len(buy_points), mode='markers', name='Buy'))
    if not sell_points.empty:
        fig.add_trace(go.Scatter(x=sell_points['timestamp'], y=[0] * len(sell_points), mode='markers', name='Sell'))
    
    return fig

def create_dashboard(df: pd.DataFrame, algorithm_results: list, symbol: str) -> go.Figure:
    """
    Create a dashboard with multiple figures arranged in a grid layout.
    
    Args:
        df (pd.DataFrame): OHLCV data
        algorithm_results (list): List of dictionaries containing algorithm results
            Each dict should have: 'name', 'buy_points', 'sell_points', 'portfolio', 'trades'
        symbol (str): Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        go.Figure: Dashboard with multiple figures
    """
    num_algorithms = len(algorithm_results)
    
    # Create subplot grid: price chart on top, then one row per algorithm with signals and portfolio
    rows = 1 + num_algorithms
    
    # Create a list of subplot titles
    subplot_titles = [f"{symbol} Price Chart"]

    subplot_titles.extend([f"{alg['name']} Signals. Change: {(alg['portfolio'].iloc[-1]['total_value'] - alg['portfolio'].iloc[0]['total_value']) / alg['portfolio'].iloc[0]['total_value']:.2%}" for alg in algorithm_results])
    
    # Create a 2D list for specs with the correct dimensions
    specs = []
    # First row for price chart
    specs.append([{"secondary_y": True}])
    # Additional rows for algorithm signals
    for _ in range(num_algorithms):
        specs.append([{"secondary_y": False}])
    
    fig = make_subplots(
        rows=rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=[0.3] + [0.7/num_algorithms] * num_algorithms,
        specs=specs
    )
    
    # Add price chart (candlesticks)
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'], open=df['open'], high=df['high'], 
            low=df['low'], close=df['close'], name='Price'
        ), 
        row=1, col=1
    )
    
    # Add algorithm-specific charts
    for i, alg in enumerate(algorithm_results):
        row_idx = i + 2  # Start from row 2 (after price chart)
        
        # Add signals chart
        if not alg['buy_points'].empty:
            fig.add_trace(
                go.Scatter(
                    x=alg['buy_points']['timestamp'], 
                    y=[0] * len(alg['buy_points']), 
                    mode='markers', 
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name=f"{alg['name']} Buy"
                ),
                row=row_idx, col=1
            )
        
        if not alg['sell_points'].empty:
            fig.add_trace(
                go.Scatter(
                    x=alg['sell_points']['timestamp'], 
                    y=[0] * len(alg['sell_points']), 
                    mode='markers', 
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name=f"{alg['name']} Sell"
                ),
                row=row_idx, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text=f"{symbol} Trading Algorithm Comparison",
        height=100 + 200 * num_algorithms,  # Adjust height based on number of algorithms
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False
    )
    
    # Hide y-axis for signal charts
    for i in range(num_algorithms):
        fig.update_yaxes(visible=False, row=i+2, col=1)
    
    return fig

def compare_algorithms(df: pd.DataFrame, algorithms: list, symbol: str, initial_cash: float, num_signals: int) -> go.Figure:
    """
    Run multiple trading algorithms and create a dashboard comparing their performance.
    
    Args:
        df (pd.DataFrame): OHLCV data
        algorithms (list): List of algorithm dictionaries with 'name' and 'trading_fn'
        symbol (str): Trading pair (e.g., 'BTC/USDT')
        initial_cash (float): Initial cash amount
        num_signals (int): Number of signals to generate
        
    Returns:
        go.Figure: Dashboard comparing algorithm performance
    """
    from portfolio.calculate import calculate_portfolio_value
    
    algorithm_results = []
    
    for alg in algorithms:
        # Generate signals
        buy_points, sell_points = alg['trading_fn'](df, num_signals)
        
        # Calculate portfolio value
        portfolio, trades = calculate_portfolio_value(df, buy_points, sell_points, initial_cash)
        
        # Store results
        algorithm_results.append({
            'name': alg['name'],
            'buy_points': buy_points,
            'sell_points': sell_points,
            'portfolio': portfolio,
            'trades': trades
        })
    
    # Create dashboard
    dashboard = create_dashboard(df, algorithm_results, symbol)
    
    return dashboard
