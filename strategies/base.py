from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


class Strategy(ABC):
    """
    Base strategy class that all strategies should inherit from.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration parameters.
        
        Args:
            config: Dictionary containing strategy parameters
        """
        self.name = self.__class__.__name__
        self.config = config
        
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate buy and sell signals based on the strategy.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        pass
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data by calculating indicators specific to the strategy.
        
        Args:
            df: OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        return df.copy()
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        return True
    
    def can_trade(self, df: pd.DataFrame, current_idx: int) -> bool:
        """
        Determine if the strategy can trade at the current index.
        
        Args:
            df: OHLCV data
            current_idx: Current index in the DataFrame
            
        Returns:
            True if the strategy can trade, False otherwise
        """
        return True
        
    def update_parameters(self, new_config: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            new_config: Dictionary containing new parameter values
        """
        self.config.update(new_config)
        
    def get_name(self) -> str:
        """
        Get strategy name.
        
        Returns:
            Strategy name
        """
        return self.name