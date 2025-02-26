import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from .base import Strategy


class MeanReversion(Strategy):
    """
    Mean Reversion strategy implementation.
    
    Uses indicators like RSI and Bollinger Bands to identify oversold/overbought conditions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MeanReversion strategy with configuration parameters.
        
        Args:
            config: Dictionary containing strategy parameters
                - rsi_period: Period for RSI calculation
                - rsi_oversold: Threshold for oversold condition
                - rsi_overbought: Threshold for overbought condition
                - bb_period: Period for Bollinger Bands
                - bb_std: Standard deviation multiplier for Bollinger Bands
                - use_rsi: Whether to use RSI for signals
                - use_bb: Whether to use Bollinger Bands for signals
                - confirmation_candles: Number of candles for confirmation
        """
        super().__init__(config)
        
        # Set default parameters if not provided
        self.rsi_period = self.config.get('rsi_period', 14)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.use_rsi = self.config.get('use_rsi', True)
        self.use_bb = self.config.get('use_bb', True)
        self.confirmation_candles = self.config.get('confirmation_candles', 1)
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion indicators and add them to the DataFrame.
        
        Args:
            df: OHLCV data
            
        Returns:
            DataFrame with added mean reversion indicators
        """
        data = df.copy()
        
        # Calculate RSI
        if self.use_rsi:
            data = self._calculate_rsi(data)
        
        # Calculate Bollinger Bands
        if self.use_bb:
            data = self._calculate_bollinger_bands(data)
            
        return data
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Relative Strength Index (RSI).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with RSI indicator
        """
        df = df.copy()
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Create gain and loss series
        df['gain'] = np.where(df['price_change'] > 0, df['price_change'], 0)
        df['loss'] = np.where(df['price_change'] < 0, -df['price_change'], 0)
        
        # Calculate average gain and loss
        df['avg_gain'] = df['gain'].rolling(window=self.rsi_period).mean()
        df['avg_loss'] = df['loss'].rolling(window=self.rsi_period).mean()
        
        # Calculate RS and RSI
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Signal generation
        df['rsi_oversold'] = df['rsi'] < self.rsi_oversold
        df['rsi_overbought'] = df['rsi'] > self.rsi_overbought
        
        # Confirmation: RSI crossing back above oversold or below overbought
        df['rsi_buy_signal'] = (df['rsi'] > self.rsi_oversold) & (df['rsi'].shift(1) <= self.rsi_oversold)
        df['rsi_sell_signal'] = (df['rsi'] < self.rsi_overbought) & (df['rsi'].shift(1) >= self.rsi_overbought)
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands indicators
        """
        df = df.copy()
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df['close'].rolling(window=self.bb_period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.bb_std)
        
        # Signal generation
        df['bb_oversold'] = df['close'] < df['bb_lower']
        df['bb_overbought'] = df['close'] > df['bb_upper']
        
        # Confirmation: Price moving back inside the bands
        df['bb_buy_signal'] = (df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'])
        df['bb_sell_signal'] = (df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate buy and sell signals based on mean reversion indicators.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # Prepare data with indicators
        data = self.prepare_data(df)
        
        # Initialize buy/sell signal columns
        data['buy_signal'] = False
        data['sell_signal'] = False
        
        # Combine signals based on enabled indicators
        if self.use_rsi and self.use_bb:
            # Both RSI and BB should confirm for stronger signal
            data['buy_signal'] = data['rsi_buy_signal'] & data['bb_oversold']
            data['sell_signal'] = data['rsi_sell_signal'] & data['bb_overbought']
        elif self.use_rsi:
            data['buy_signal'] = data['rsi_buy_signal']
            data['sell_signal'] = data['rsi_sell_signal']
        elif self.use_bb:
            data['buy_signal'] = data['bb_buy_signal']
            data['sell_signal'] = data['bb_sell_signal']
        
        # Get buy and sell signal dataframes
        buy_signals = data[data['buy_signal']].copy()
        sell_signals = data[data['sell_signal']].copy()
        
        # Format signals
        if not buy_signals.empty:
            buy_signals = buy_signals[['timestamp', 'close']].rename(columns={'close': 'price'})
        else:
            buy_signals = pd.DataFrame(columns=['timestamp', 'price'])
        
        if not sell_signals.empty:
            sell_signals = sell_signals[['timestamp', 'close']].rename(columns={'close': 'price'})
        else:
            sell_signals = pd.DataFrame(columns=['timestamp', 'price'])
        
        return buy_signals, sell_signals
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        if self.rsi_period <= 0:
            print("Error: rsi_period must be positive")
            return False
            
        if self.rsi_oversold >= self.rsi_overbought:
            print("Error: rsi_oversold must be less than rsi_overbought")
            return False
            
        if self.bb_period <= 0:
            print("Error: bb_period must be positive")
            return False
            
        if self.bb_std <= 0:
            print("Error: bb_std must be positive")
            return False
            
        if not (self.use_rsi or self.use_bb):
            print("Error: at least one indicator (RSI or BB) must be enabled")
            return False
            
        return True