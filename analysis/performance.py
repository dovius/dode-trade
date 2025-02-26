import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class PerformanceAnalyzer:
    """
    Analyzes trading strategy performance.
    
    Calculates key performance metrics and generates performance reports.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PerformanceAnalyzer with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Create results directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def calculate_metrics(self, portfolio: pd.DataFrame, trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from portfolio and trades data.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            trades: DataFrame with individual trade data
            
        Returns:
            Dictionary with performance metrics
        """
        # Check if we have data
        if portfolio.empty or trades.empty:
            return {
                'error': 'No portfolio or trade data available'
            }
        
        # Extract key values
        initial_equity = portfolio['total_value'].iloc[0]
        final_equity = portfolio['total_value'].iloc[-1]
        
        # Calculate returns
        total_return = (final_equity / initial_equity - 1) * 100
        
        # Calculate drawdown
        portfolio['high_water_mark'] = portfolio['total_value'].cummax()
        portfolio['drawdown'] = (portfolio['high_water_mark'] - portfolio['total_value']) / portfolio['high_water_mark'] * 100
        max_drawdown = portfolio['drawdown'].max()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        daily_returns = portfolio['total_value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 else 0
        
        # Calculate trade metrics
        num_trades = len(trades)
        winning_trades = trades[trades['profit_percentage'] > 0]
        losing_trades = trades[trades['profit_percentage'] <= 0]
        
        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0
        
        avg_profit = winning_trades['profit_percentage'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['profit_percentage'].mean() if len(losing_trades) > 0 else 0
        
        # Calculate profit factor
        total_profit = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (win_rate / 100 * avg_profit) + ((100 - win_rate) / 100 * avg_loss)
        
        # Calculate maximum consecutive wins/losses
        if not trades.empty:
            trades['win'] = trades['profit_percentage'] > 0
            
            # Calculate streaks
            trades['streak'] = (trades['win'] != trades['win'].shift(1)).cumsum()
            win_streaks = trades[trades['win']].groupby('streak').size()
            loss_streaks = trades[~trades['win']].groupby('streak').size()
            
            max_win_streak = win_streaks.max() if not win_streaks.empty else 0
            max_loss_streak = loss_streaks.max() if not loss_streaks.empty else 0
        else:
            max_win_streak = 0
            max_loss_streak = 0
        
        # Return all calculated metrics
        return {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'number_of_trades': num_trades,
            'win_rate_pct': win_rate,
            'avg_profit_pct': avg_profit,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'total_profit': total_profit,
            'total_loss': total_loss
        }
    
    def compare_strategies(self, strategy_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare performance metrics of multiple strategies.
        
        Args:
            strategy_results: Dictionary with strategy names as keys and performance metrics as values
            
        Returns:
            DataFrame with comparative metrics
        """
        # Create a DataFrame for comparison
        comparison = pd.DataFrame()
        
        for strategy_name, metrics in strategy_results.items():
            # Create a Series with the metrics
            series = pd.Series(metrics, name=strategy_name)
            
            # Add to comparison DataFrame
            comparison[strategy_name] = series
        
        # Transpose for better readability
        comparison = comparison.T
        
        # Select key metrics for comparison
        key_metrics = [
            'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio',
            'win_rate_pct', 'profit_factor', 'expectancy'
        ]
        
        selected_metrics = comparison[key_metrics]
        
        return selected_metrics
    
    def plot_equity_curve(self, portfolio: pd.DataFrame, trades: pd.DataFrame = None, 
                         title: str = 'Equity Curve', save_path: Optional[str] = None) -> None:
        """
        Plot equity curve with optional trade markers.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            trades: DataFrame with trades (optional)
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.plot(portfolio['timestamp'], portfolio['total_value'], label='Equity')
        
        # Add drawdown as area
        portfolio['high_water_mark'] = portfolio['total_value'].cummax()
        portfolio['drawdown'] = portfolio['high_water_mark'] - portfolio['total_value']
        plt.fill_between(portfolio['timestamp'], 
                        portfolio['high_water_mark'], 
                        portfolio['total_value'], 
                        alpha=0.3, color='red', label='Drawdown')
        
        # Add trade markers if available
        if trades is not None and not trades.empty:
            # Buy points
            buy_trades = trades[trades['buy_price'] > 0]
            if not buy_trades.empty:
                plt.scatter(buy_trades['buy_time'], 
                           [portfolio.loc[portfolio['timestamp'] == t, 'total_value'].iloc[0] 
                            if t in portfolio['timestamp'].values else np.nan 
                            for t in buy_trades['buy_time']], 
                           marker='^', color='green', s=100, label='Buy')
            
            # Sell points
            sell_trades = trades[trades['sell_price'] > 0]
            if not sell_trades.empty:
                plt.scatter(sell_trades['sell_time'], 
                           [portfolio.loc[portfolio['timestamp'] == t, 'total_value'].iloc[0] 
                            if t in portfolio['timestamp'].values else np.nan 
                            for t in sell_trades['sell_time']], 
                           marker='v', color='red', s=100, label='Sell')
        
        # Add formatting
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Equity', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_drawdown_chart(self, portfolio: pd.DataFrame, 
                          title: str = 'Drawdown Analysis', save_path: Optional[str] = None) -> None:
        """
        Plot drawdown chart.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(12, 6))
        
        # Calculate drawdown percentage
        portfolio['high_water_mark'] = portfolio['total_value'].cummax()
        portfolio['drawdown_pct'] = (portfolio['high_water_mark'] - portfolio['total_value']) / portfolio['high_water_mark'] * 100
        
        # Plot drawdown
        plt.plot(portfolio['timestamp'], portfolio['drawdown_pct'])
        plt.fill_between(portfolio['timestamp'], 0, portfolio['drawdown_pct'], alpha=0.3, color='red')
        
        # Add formatting
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_monthly_returns(self, portfolio: pd.DataFrame, 
                           title: str = 'Monthly Returns', save_path: Optional[str] = None) -> None:
        """
        Plot monthly returns heatmap.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        # Calculate daily returns
        portfolio['daily_return'] = portfolio['total_value'].pct_change()
        
        # Extract year and month
        portfolio['year'] = portfolio['timestamp'].dt.year
        portfolio['month'] = portfolio['timestamp'].dt.month
        
        # Calculate monthly returns
        monthly_returns = portfolio.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create pivot table
        pivot_returns = monthly_returns.pivot(index='year', columns='month', values='daily_return')
        
        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_returns * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   linewidths=1, cbar_kws={'label': 'Return (%)'})
        
        # Add month names
        plt.xticks(np.arange(12) + 0.5, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Add formatting
        plt.title(title, fontsize=16)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_performance_report(self, portfolio: pd.DataFrame, trades: pd.DataFrame, 
                                  strategy_name: str = 'Strategy', save_dir: str = 'results') -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            portfolio: DataFrame with portfolio values over time
            trades: DataFrame with trades
            strategy_name: Name of the strategy
            save_dir: Directory to save report files
            
        Returns:
            Dictionary with performance metrics
        """
        # Create directory for strategy if it doesn't exist
        strategy_dir = os.path.join(save_dir, strategy_name.replace(' ', '_'))
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(portfolio, trades)
        
        # Save metrics to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = os.path.join(strategy_dir, f'metrics_{timestamp}.json')
        pd.Series(metrics).to_json(metrics_path)
        
        # Generate and save plots
        self.plot_equity_curve(
            portfolio, trades, 
            title=f'{strategy_name} - Equity Curve',
            save_path=os.path.join(strategy_dir, f'equity_curve_{timestamp}.png')
        )
        
        self.plot_drawdown_chart(
            portfolio,
            title=f'{strategy_name} - Drawdown Analysis',
            save_path=os.path.join(strategy_dir, f'drawdown_{timestamp}.png')
        )
        
        self.plot_monthly_returns(
            portfolio,
            title=f'{strategy_name} - Monthly Returns',
            save_path=os.path.join(strategy_dir, f'monthly_returns_{timestamp}.png')
        )
        
        # Create trade analysis plots
        if not trades.empty:
            # Profit distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(trades['profit_percentage'], kde=True)
            plt.title(f'{strategy_name} - Profit Distribution', fontsize=16)
            plt.xlabel('Profit (%)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(strategy_dir, f'profit_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Win/Loss ratio
            plt.figure(figsize=(8, 8))
            win_count = len(trades[trades['profit_percentage'] > 0])
            loss_count = len(trades) - win_count
            plt.pie([win_count, loss_count], labels=['Win', 'Loss'], autopct='%1.1f%%', colors=['green', 'red'])
            plt.title(f'{strategy_name} - Win/Loss Ratio', fontsize=16)
            plt.savefig(os.path.join(strategy_dir, f'win_loss_ratio_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return metrics
    
    def generate_strategy_comparison_report(self, strategies_results: Dict[str, Dict[str, Any]], 
                                         save_dir: str = 'results') -> None:
        """
        Generate comparative report for multiple strategies.
        
        Args:
            strategies_results: Dictionary with strategy names as keys and performance metrics as values
            save_dir: Directory to save report files
        """
        if not strategies_results:
            return
        
        # Create comparison DataFrame
        comparison = self.compare_strategies(strategies_results)
        
        # Save comparison to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = os.path.join(save_dir, f'strategy_comparison_{timestamp}.csv')
        comparison.to_csv(comparison_path)
        
        # Generate comparison plots
        key_metrics = ['total_return_pct', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate_pct']
        
        # Bar chart comparison
        plt.figure(figsize=(14, 10))
        comparison[key_metrics].plot(kind='bar', subplots=True, layout=(2, 2), sharex=False, figsize=(14, 10))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'strategy_comparison_bars_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Radar chart comparison (Spider chart)
        # Normalize values for radar chart
        radar_data = comparison[key_metrics].copy()
        for col in radar_data.columns:
            if col == 'max_drawdown_pct':
                # Invert drawdown (lower is better)
                radar_data[col] = (radar_data[col].max() - radar_data[col]) / (radar_data[col].max() - radar_data[col].min())
            else:
                radar_data[col] = (radar_data[col] - radar_data[col].min()) / (radar_data[col].max() - radar_data[col].min())
        
        # Create radar chart
        plt.figure(figsize=(10, 10))
        
        # Number of variables
        categories = key_metrics
        N = len(categories)
        
        # Create angle for each variable
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize the plot
        ax = plt.subplot(111, polar=True)
        
        # Add each strategy
        for idx, strategy in enumerate(radar_data.index):
            values = radar_data.loc[strategy].tolist()
            values += values[:1]  # Close the loop
            
            # Plot
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy)
            ax.fill(angles, values, alpha=0.1)
        
        # Set display to show all variables
        plt.xticks(angles[:-1], categories)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.title('Strategy Comparison Radar Chart', size=20)
        plt.savefig(os.path.join(save_dir, f'strategy_comparison_radar_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()