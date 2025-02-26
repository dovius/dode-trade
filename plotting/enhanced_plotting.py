import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


class EnhancedPlotter:
    """
    Enhanced plotting functionality for trading strategies.
    
    Generates interactive charts with strategy indicators and performance metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize EnhancedPlotter with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.show_indicators = config.get('show_all_indicators', True)
        self.save_charts = config.get('save_charts', True)
        self.charts_dir = config.get('charts_dir', 'charts')
        
        # Create charts directory if needed
        if self.save_charts and not os.path.exists(self.charts_dir):
            os.makedirs(self.charts_dir)
    
    def plot_strategy_backtest(self, df: pd.DataFrame, strategy_data: Dict[str, pd.DataFrame], 
                             buy_points: pd.DataFrame, sell_points: pd.DataFrame, 
                             portfolio: pd.DataFrame, trades_df: Optional[pd.DataFrame] = None,
                             strategy_name: str = 'Strategy', 
                             symbol: str = 'BTC/USDT') -> go.Figure:
        """
        Generate comprehensive backtest visualization with indicators.
        
        Args:
            df: OHLCV data
            strategy_data: Dictionary of DataFrames with strategy indicators
            buy_points: Buy signals
            sell_points: Sell signals
            portfolio: Portfolio value data
            trades_df: Trade performance data (optional)
            strategy_name: Strategy name
            symbol: Trading pair symbol
            
        Returns:
            Plotly Figure object
        """
        # Count indicators to determine number of rows
        num_indicator_rows = sum(1 for _, indicators_df in strategy_data.items() if not indicators_df.empty)
        
        # Create subplots based on available data
        fig = make_subplots(
            rows=3 + num_indicator_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f"{symbol} Price Chart with {strategy_name} Signals",
                "Portfolio Value",
                "Trade Profitability",
                *[f"{name} Indicators" for name in strategy_data.keys() if not strategy_data[name].empty]
            ),
            row_heights=[0.4, 0.2, 0.2, *[0.2] * num_indicator_rows]
        )
        
        # Price chart with candlesticks and volume
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
        
        # Add volume as bar chart
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['volume'],
                name='Volume',
                marker_color='rgba(0,0,0,0.3)'
            ),
            row=1, col=1
        )
        
        # Add buy and sell signals as markers
        if not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points['timestamp'],
                    y=buy_points['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    )
                ),
                row=1, col=1
            )
            
        if not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points['timestamp'],
                    y=sell_points['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red',
                        line=dict(width=2, color='darkred')
                    )
                ),
                row=1, col=1
            )
        
        # Portfolio value chart
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=portfolio['cash'],
                name='Cash',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=portfolio['crypto_value'],
                name='Crypto Value',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=portfolio['total_value'],
                name='Total Value',
                line=dict(color='blue', width=2)
            ),
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
        
        # Add strategy indicators
        current_row = 4
        for strategy_name, indicators_df in strategy_data.items():
            if indicators_df.empty:
                continue
                
            # Determine which columns are indicators (excluding basic OHLCV)
            basic_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            indicator_cols = [col for col in indicators_df.columns if col not in basic_cols]
            
            # Add each indicator
            for col in indicator_cols:
                # Skip columns that are boolean or don't look like indicators
                if indicators_df[col].dtype == bool or col.endswith('_signal') or col.endswith('_change'):
                    continue
                    
                # Determine if this is a line or scatter plot
                mode = 'lines'
                
                # For oscillators (like RSI), set color based on thresholds
                color = 'blue'
                line_width = 1
                
                # Special case for certain indicators
                if col == 'short_ma' or col == 'sma':
                    color = 'orange'
                    line_width = 2
                elif col == 'long_ma' or col == 'ema':
                    color = 'purple'
                    line_width = 2
                elif col == 'rsi':
                    # Add threshold lines for RSI
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df['timestamp'],
                            y=[70] * len(indicators_df),
                            mode='lines',
                            line=dict(color='red', width=1, dash='dash'),
                            name='RSI Overbought'
                        ),
                        row=current_row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=indicators_df['timestamp'],
                            y=[30] * len(indicators_df),
                            mode='lines',
                            line=dict(color='green', width=1, dash='dash'),
                            name='RSI Oversold'
                        ),
                        row=current_row, col=1
                    )
                    
                    color = 'blue'
                    line_width = 2
                elif col.startswith('bb_'):
                    if col == 'bb_upper':
                        color = 'gray'
                    elif col == 'bb_lower':
                        color = 'gray'
                    elif col == 'bb_middle':
                        color = 'blue'
                        line_width = 2
                    
                # Add the indicator trace
                fig.add_trace(
                    go.Scatter(
                        x=indicators_df['timestamp'],
                        y=indicators_df[col],
                        mode=mode,
                        name=col,
                        line=dict(color=color, width=line_width)
                    ),
                    row=current_row, col=1
                )
            
            current_row += 1
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} - {strategy_name} Backtest Analysis",
            height=300 * (3 + num_indicator_rows),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
        fig.update_yaxes(title_text="Profit/Loss (%)", row=3, col=1)
        
        # Update indicator axis labels
        current_row = 4
        for strategy_name, indicators_df in strategy_data.items():
            if not indicators_df.empty:
                # Determine y-axis label based on indicators
                if 'rsi' in indicators_df.columns:
                    y_title = "RSI"
                elif 'short_ma' in indicators_df.columns and 'long_ma' in indicators_df.columns:
                    y_title = "Moving Averages"
                elif 'bb_upper' in indicators_df.columns:
                    y_title = "Bollinger Bands"
                else:
                    y_title = "Indicators"
                    
                fig.update_yaxes(title_text=y_title, row=current_row, col=1)
                current_row += 1
        
        # Save chart if configured
        if self.save_charts:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('/', '_')}_{strategy_name.replace(' ', '_')}_{timestamp}.html"
            fig.write_html(os.path.join(self.charts_dir, filename))
        
        return fig
    
    def plot_multi_strategy_comparison(self, df: pd.DataFrame, 
                                    strategies_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """
        Generate comparison visualization for multiple strategies.
        
        Args:
            df: OHLCV data
            strategies_results: Dictionary with strategy names as keys and results as values
                Each strategy result should contain:
                - 'portfolio': DataFrame with portfolio values
                - 'trades': DataFrame with trades
                - 'buy_points': DataFrame with buy signals
                - 'sell_points': DataFrame with sell signals
                - 'metrics': Dictionary with performance metrics
            
        Returns:
            Plotly Figure object
        """
        # Create subplot structure
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Price Chart with Strategy Signals", "Strategy Performance Comparison"),
            row_heights=[0.6, 0.4]
        )
        
        # Plot price data
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
        
        # Define colors for different strategies
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
        # Plot each strategy's signals and performance
        for i, (strategy_name, results) in enumerate(strategies_results.items()):
            color = colors[i % len(colors)]
            
            # Plot buy signals
            buy_points = results.get('buy_points', pd.DataFrame())
            if not buy_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_points['timestamp'],
                        y=buy_points['price'],
                        mode='markers',
                        name=f'{strategy_name} Buy',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color=color,
                            line=dict(width=1, color='darkgreen')
                        )
                    ),
                    row=1, col=1
                )
            
            # Plot sell signals
            sell_points = results.get('sell_points', pd.DataFrame())
            if not sell_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_points['timestamp'],
                        y=sell_points['price'],
                        mode='markers',
                        name=f'{strategy_name} Sell',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color=color,
                            line=dict(width=1, color='darkred')
                        )
                    ),
                    row=1, col=1
                )
            
            # Plot portfolio equity curve
            portfolio = results.get('portfolio', pd.DataFrame())
            if not portfolio.empty:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=portfolio['total_value'],
                        mode='lines',
                        name=f'{strategy_name} Equity',
                        line=dict(color=color, width=2)
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title="Strategy Comparison Analysis",
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
        )
        
        # Save chart if configured
        if self.save_charts:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else "COMPARISON"
            filename = f"{symbol.replace('/', '_')}_strategy_comparison_{timestamp}.html"
            fig.write_html(os.path.join(self.charts_dir, filename))
        
        return fig
    
    def plot_trade_analysis(self, trades_df: pd.DataFrame, strategy_name: str = 'Strategy') -> Dict[str, go.Figure]:
        """
        Generate trade analysis visualizations.
        
        Args:
            trades_df: DataFrame with trade data
            strategy_name: Strategy name
            
        Returns:
            Dictionary of Plotly Figure objects
        """
        figures = {}
        
        if trades_df.empty:
            return figures
        
        # Trade distribution
        fig_dist = go.Figure()
        
        fig_dist.add_trace(
            go.Histogram(
                x=trades_df['profit_percentage'],
                name='Profit Distribution',
                xbins=dict(size=1),
                marker_color='blue',
                opacity=0.7
            )
        )
        
        # Add kernel density estimate
        trade_profits = trades_df['profit_percentage'].dropna()
        if len(trade_profits) > 5:  # Need sufficient data for KDE
            x_range = np.linspace(trade_profits.min(), trade_profits.max(), 100)
            
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(trade_profits)
            y_range = kde(x_range)
            
            # Scale KDE to match histogram
            scaling_factor = max(np.histogram(trade_profits, bins=20)[0]) / max(y_range)
            
            fig_dist.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range * scaling_factor,
                    mode='lines',
                    name='Density',
                    line=dict(color='red', width=2)
                )
            )
        
        fig_dist.update_layout(
            title=f"{strategy_name} - Trade Profit Distribution",
            xaxis_title="Profit Percentage (%)",
            yaxis_title="Frequency",
            bargap=0.1
        )
        
        figures['distribution'] = fig_dist
        
        # Win/Loss pie chart
        win_count = len(trades_df[trades_df['profit_percentage'] > 0])
        loss_count = len(trades_df) - win_count
        
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=['Win', 'Loss'],
                values=[win_count, loss_count],
                textinfo='percent+label',
                marker=dict(colors=['green', 'red'])
            )
        ])
        
        fig_pie.update_layout(
            title=f"{strategy_name} - Win/Loss Ratio"
        )
        
        figures['win_loss'] = fig_pie
        
        # Profit over time
        trades_df_sorted = trades_df.sort_values('sell_time')
        trades_df_sorted['cumulative_profit'] = trades_df_sorted['profit_loss'].cumsum()
        
        fig_profit = go.Figure()
        
        fig_profit.add_trace(
            go.Scatter(
                x=trades_df_sorted['sell_time'],
                y=trades_df_sorted['cumulative_profit'],
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='blue', width=2)
            )
        )
        
        fig_profit.update_layout(
            title=f"{strategy_name} - Cumulative Profit Over Time",
            xaxis_title="Date",
            yaxis_title="Profit"
        )
        
        figures['profit_time'] = fig_profit
        
        # Save charts if configured
        if self.save_charts:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_strategy_name = strategy_name.replace(' ', '_')
            
            for name, fig in figures.items():
                filename = f"{safe_strategy_name}_trade_{name}_{timestamp}.html"
                fig.write_html(os.path.join(self.charts_dir, filename))
        
        return figures