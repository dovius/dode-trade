import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from .base import Strategy


class TrendFollowing(Strategy):
    """
    Trend Following strategy implementation.
    
    Uses moving average crossovers and other trend indicators to generate signals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TrendFollowing strategy with configuration parameters.
        
        Args:
            config: Dictionary containing strategy parameters
                - short_window: Period for short moving average
                - long_window: Period for long moving average
                - adx_period: Period for ADX calculation
                - adx_threshold: Threshold for trend strength
                - use_adx_filter: Whether to use ADX as a filter
        """
        super().__init__(config)
        
        # Set default parameters if not provided
        self.short_window = self.config.get('short_window', 9)
        self.long_window = self.config.get('long_window', 21)
        self.adx_period = self.config.get('adx_period', 14)
        self.adx_threshold = self.config.get('adx_threshold', 25)
        self.use_adx_filter = self.config.get('use_adx_filter', True)
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend indicators and add them to the DataFrame.
        
        Args:
            df: OHLCV data
            
        Returns:
            DataFrame with added trend indicators
        """
        data = df.copy()
        
        # Calculate short and long moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window, min_periods=1).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Calculate trend signal (1 = uptrend, 0 = downtrend)
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        
        # Calculate position change (1 = buy, -1 = sell)
        data['position_change'] = data['signal'].diff()
        
        # Calculate ADX for trend strength if enabled
        if self.use_adx_filter:
            data = self._calculate_adx(data)
        
        return data
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the Average Directional Index (ADX) to measure trend strength.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ADX indicator
        """
        df = df.copy()
        
        # Calculate +DI and -DI
        df['high_delta'] = df['high'] - df['high'].shift(1)
        df['low_delta'] = df['low'].shift(1) - df['low']
        
        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.adx_period).mean()
        
        # +DM and -DM
        df['plus_dm'] = 0
        df['minus_dm'] = 0
        df.loc[(df['high_delta'] > df['low_delta']) & (df['high_delta'] > 0), 'plus_dm'] = df['high_delta']
        df.loc[(df['low_delta'] > df['high_delta']) & (df['low_delta'] > 0), 'minus_dm'] = df['low_delta']
        
        # Smoothed +DM and -DM
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=self.adx_period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=self.adx_period).mean() / df['atr'])
        
        # Directional Index
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # Average Directional Index
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate buy and sell signals based on trend following.
        
        Args:
            df: OHLCV data
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # Prepare data with indicators
        data = self.prepare_data(df)
        
        # Apply ADX filter if enabled
        if self.use_adx_filter:
            # Strong trend filter (ADX > threshold)
            strong_trend = data['adx'] > self.adx_threshold
            
            # Buy when short MA crosses above long MA AND trend is strong
            buy_signals = data[(data['position_change'] == 1) & strong_trend].copy()
            
            # Sell when short MA crosses below long MA
            sell_signals = data[data['position_change'] == -1].copy()
        else:
            # Buy when short MA crosses above long MA
            buy_signals = data[data['position_change'] == 1].copy()
            
            # Sell when short MA crosses below long MA
            sell_signals = data[data['position_change'] == -1].copy()
        
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
        if self.short_window >= self.long_window:
            print("Error: short_window must be less than long_window")
            return False
        
        if self.short_window <= 0 or self.long_window <= 0:
            print("Error: window sizes must be positive")
            return False
            
        return True