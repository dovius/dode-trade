import pandas as pd
import importlib
import logging
from typing import Dict, Any, List, Tuple, Optional
from .base import Strategy

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages multiple trading strategies.
    
    Handles strategy initialization, signal aggregation, and performance tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StrategyManager with configuration parameters.
        
        Args:
            config: Dictionary containing configuration parameters
                - strategies: List of strategy configurations
                    - name: Strategy name
                    - module: Strategy module path
                    - class: Strategy class name
                    - weight: Strategy weight for signal aggregation
                    - parameters: Strategy-specific parameters
                - enable_aggregation: Whether to aggregate signals from multiple strategies
                - aggregation_method: Method to aggregate signals ('weighted', 'vote', 'priority')
        """
        self.config = config
        self.strategies = {}
        self.strategy_weights = {}
        self.strategy_performance = {}
        
        # Set default parameters
        self.enable_aggregation = self.config.get('enable_aggregation', False)
        self.aggregation_method = self.config.get('aggregation_method', 'weighted')
        
        # Load strategies
        self._load_strategies()
        
    def _load_strategies(self) -> None:
        """
        Load and initialize strategies from configuration.
        """
        strategies_config = self.config.get('strategies', [])
        
        for strategy_config in strategies_config:
            name = strategy_config.get('name')
            module_path = strategy_config.get('module')
            class_name = strategy_config.get('class')
            weight = strategy_config.get('weight', 1.0)
            parameters = strategy_config.get('parameters', {})
            
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Get the strategy class
                strategy_class = getattr(module, class_name)
                
                # Initialize the strategy
                strategy = strategy_class(parameters)
                
                # Validate strategy parameters
                if not strategy.validate_parameters():
                    logger.error(f"Invalid parameters for strategy {name}")
                    continue
                
                # Store the strategy and its weight
                self.strategies[name] = strategy
                self.strategy_weights[name] = weight
                self.strategy_performance[name] = {
                    'signals_generated': 0,
                    'successful_trades': 0,
                    'failed_trades': 0,
                    'total_profit': 0.0,
                    'total_loss': 0.0,
                    'win_rate': 0.0
                }
                
                logger.info(f"Loaded strategy: {name}")
                
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
                
    def get_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get trading signals from all strategies.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        if not self.strategies:
            logger.warning("No strategies loaded")
            return pd.DataFrame(columns=['timestamp', 'price']), pd.DataFrame(columns=['timestamp', 'price'])
            
        if self.enable_aggregation:
            return self._get_aggregated_signals(df)
        else:
            # Get signals from the first strategy (if only one is configured)
            strategy_name = list(self.strategies.keys())[0]
            strategy = self.strategies[strategy_name]
            buy_signals, sell_signals = strategy.generate_signals(df)
            
            # Track the number of signals generated
            self.strategy_performance[strategy_name]['signals_generated'] += len(buy_signals) + len(sell_signals)
            
            return buy_signals, sell_signals
            
    def _get_aggregated_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get aggregated signals from multiple strategies.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        all_buy_signals = {}
        all_sell_signals = {}
        
        # Get signals from each strategy
        for name, strategy in self.strategies.items():
            buy_signals, sell_signals = strategy.generate_signals(df)
            
            # Track the number of signals generated
            self.strategy_performance[name]['signals_generated'] += len(buy_signals) + len(sell_signals)
            
            # Store signals
            all_buy_signals[name] = buy_signals
            all_sell_signals[name] = sell_signals
            
        # Aggregate signals based on the selected method
        if self.aggregation_method == 'weighted':
            return self._aggregate_weighted_signals(all_buy_signals, all_sell_signals)
        elif self.aggregation_method == 'vote':
            return self._aggregate_vote_signals(all_buy_signals, all_sell_signals, df)
        elif self.aggregation_method == 'priority':
            return self._aggregate_priority_signals(all_buy_signals, all_sell_signals)
        else:
            logger.warning(f"Unknown aggregation method: {self.aggregation_method}")
            return pd.DataFrame(columns=['timestamp', 'price']), pd.DataFrame(columns=['timestamp', 'price'])
            
    def _aggregate_weighted_signals(self, all_buy_signals: Dict[str, pd.DataFrame], all_sell_signals: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate signals using the weighted method.
        
        The signal strength is proportional to the strategy weight.
        
        Args:
            all_buy_signals: Dictionary of buy signals from each strategy
            all_sell_signals: Dictionary of sell signals from each strategy
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # Combine all buy signals
        buy_signals_list = []
        for name, signals in all_buy_signals.items():
            if not signals.empty:
                signals_copy = signals.copy()
                signals_copy['strategy'] = name
                signals_copy['weight'] = self.strategy_weights[name]
                buy_signals_list.append(signals_copy)
                
        # Combine all sell signals
        sell_signals_list = []
        for name, signals in all_sell_signals.items():
            if not signals.empty:
                signals_copy = signals.copy()
                signals_copy['strategy'] = name
                signals_copy['weight'] = self.strategy_weights[name]
                sell_signals_list.append(signals_copy)
                
        # Merge signals
        if buy_signals_list:
            buy_signals = pd.concat(buy_signals_list)
            
            # Aggregate signals at the same timestamp
            buy_signals = buy_signals.groupby('timestamp').agg({
                'price': 'mean',
                'weight': 'sum'
            }).reset_index()
            
            # Keep only signals with sufficient weight
            buy_signals = buy_signals[buy_signals['weight'] > 0.5]
        else:
            buy_signals = pd.DataFrame(columns=['timestamp', 'price'])
            
        if sell_signals_list:
            sell_signals = pd.concat(sell_signals_list)
            
            # Aggregate signals at the same timestamp
            sell_signals = sell_signals.groupby('timestamp').agg({
                'price': 'mean',
                'weight': 'sum'
            }).reset_index()
            
            # Keep only signals with sufficient weight
            sell_signals = sell_signals[sell_signals['weight'] > 0.5]
        else:
            sell_signals = pd.DataFrame(columns=['timestamp', 'price'])
            
        return buy_signals[['timestamp', 'price']], sell_signals[['timestamp', 'price']]
        
    def _aggregate_vote_signals(self, all_buy_signals: Dict[str, pd.DataFrame], all_sell_signals: Dict[str, pd.DataFrame], df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate signals using the vote method.
        
        A signal is generated if a majority of strategies agree.
        
        Args:
            all_buy_signals: Dictionary of buy signals from each strategy
            all_sell_signals: Dictionary of sell signals from each strategy
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # Get all unique timestamps
        all_timestamps = set()
        for signals in all_buy_signals.values():
            all_timestamps.update(signals['timestamp'].tolist())
        for signals in all_sell_signals.values():
            all_timestamps.update(signals['timestamp'].tolist())
            
        all_timestamps = sorted(all_timestamps)
        
        # Count votes for each timestamp
        buy_votes = {}
        sell_votes = {}
        
        for timestamp in all_timestamps:
            buy_votes[timestamp] = 0
            sell_votes[timestamp] = 0
            
            for name, signals in all_buy_signals.items():
                if timestamp in signals['timestamp'].values:
                    buy_votes[timestamp] += 1
                    
            for name, signals in all_sell_signals.items():
                if timestamp in signals['timestamp'].values:
                    sell_votes[timestamp] += 1
                    
        # Generate signals based on majority vote
        majority_threshold = len(self.strategies) / 2
        
        buy_signals_data = []
        for timestamp, votes in buy_votes.items():
            if votes > majority_threshold:
                # Get the price at this timestamp
                price = df.loc[df['timestamp'] == timestamp, 'close'].iloc[0] if timestamp in df['timestamp'].values else 0
                buy_signals_data.append({'timestamp': timestamp, 'price': price})
                
        sell_signals_data = []
        for timestamp, votes in sell_votes.items():
            if votes > majority_threshold:
                # Get the price at this timestamp
                price = df.loc[df['timestamp'] == timestamp, 'close'].iloc[0] if timestamp in df['timestamp'].values else 0
                sell_signals_data.append({'timestamp': timestamp, 'price': price})
                
        buy_signals = pd.DataFrame(buy_signals_data)
        sell_signals = pd.DataFrame(sell_signals_data)
        
        return buy_signals, sell_signals
        
    def _aggregate_priority_signals(self, all_buy_signals: Dict[str, pd.DataFrame], all_sell_signals: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate signals using the priority method.
        
        Signals from higher-priority strategies (higher weight) take precedence.
        
        Args:
            all_buy_signals: Dictionary of buy signals from each strategy
            all_sell_signals: Dictionary of sell signals from each strategy
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # Sort strategies by weight (higher weight = higher priority)
        sorted_strategies = sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True)
        
        # Take signals from the highest-priority strategy that has signals
        for name, _ in sorted_strategies:
            buy_signals = all_buy_signals.get(name, pd.DataFrame(columns=['timestamp', 'price']))
            sell_signals = all_sell_signals.get(name, pd.DataFrame(columns=['timestamp', 'price']))
            
            if not buy_signals.empty or not sell_signals.empty:
                return buy_signals, sell_signals
                
        return pd.DataFrame(columns=['timestamp', 'price']), pd.DataFrame(columns=['timestamp', 'price'])
        
    def update_performance(self, strategy_name: str, trade_result: Dict[str, Any]) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_name: Strategy name
            trade_result: Dictionary with trade result information
        """
        if strategy_name not in self.strategy_performance:
            logger.warning(f"Strategy {strategy_name} not found in performance tracking")
            return
            
        # Update performance metrics
        is_successful = trade_result.get('is_profitable', False)
        profit_loss = trade_result.get('profit_loss', 0.0)
        
        if is_successful:
            self.strategy_performance[strategy_name]['successful_trades'] += 1
            self.strategy_performance[strategy_name]['total_profit'] += profit_loss
        else:
            self.strategy_performance[strategy_name]['failed_trades'] += 1
            self.strategy_performance[strategy_name]['total_loss'] += profit_loss
            
        # Recalculate win rate
        total_trades = (self.strategy_performance[strategy_name]['successful_trades'] + 
                        self.strategy_performance[strategy_name]['failed_trades'])
        
        if total_trades > 0:
            self.strategy_performance[strategy_name]['win_rate'] = (
                self.strategy_performance[strategy_name]['successful_trades'] / total_trades * 100
            )
            
    def get_performance_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance report for all strategies.
        
        Returns:
            Dictionary with performance metrics for each strategy
        """
        return self.strategy_performance
        
    def get_strategy(self, name: str) -> Optional[Strategy]:
        """
        Get a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy object or None if not found
        """
        return self.strategies.get(name)
        
    def get_all_strategies(self) -> Dict[str, Strategy]:
        """
        Get all strategies.
        
        Returns:
            Dictionary of strategy objects
        """
        return self.strategies
        
    def add_strategy(self, name: str, strategy: Strategy, weight: float = 1.0) -> None:
        """
        Add a new strategy.
        
        Args:
            name: Strategy name
            strategy: Strategy object
            weight: Strategy weight
        """
        self.strategies[name] = strategy
        self.strategy_weights[name] = weight
        self.strategy_performance[name] = {
            'signals_generated': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"Added strategy: {name}")
        
    def remove_strategy(self, name: str) -> bool:
        """
        Remove a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            True if the strategy was removed, False otherwise
        """
        if name in self.strategies:
            del self.strategies[name]
            del self.strategy_weights[name]
            del self.strategy_performance[name]
            
            logger.info(f"Removed strategy: {name}")
            return True
        else:
            logger.warning(f"Strategy {name} not found")
            return False
            
    def update_strategy_weight(self, name: str, weight: float) -> bool:
        """
        Update the weight of a strategy.
        
        Args:
            name: Strategy name
            weight: New weight
            
        Returns:
            True if the weight was updated, False otherwise
        """
        if name in self.strategy_weights:
            self.strategy_weights[name] = weight
            logger.info(f"Updated weight for strategy {name}: {weight}")
            return True
        else:
            logger.warning(f"Strategy {name} not found")
            return False