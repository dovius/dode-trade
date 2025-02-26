import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Risk management system for trading strategies.
    
    Handles position sizing, stop-loss placement, and drawdown protection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RiskManager with configuration parameters.
        
        Args:
            config: Dictionary containing risk parameters
                - risk_per_trade: Percentage of equity to risk per trade
                - max_risk_per_day: Maximum percentage of equity to risk per day
                - max_drawdown: Maximum drawdown percentage before reducing risk
                - stop_multiplier: ATR multiplier for stop-loss placement
                - atr_period: Period for ATR calculation
                - trailing_stop: Whether to use trailing stops
                - take_profit_ratio: Risk-reward ratio for take-profit placement
                - initial_equity: Initial account equity
                - use_fixed_stops: Whether to use fixed percentage stops
                - fixed_stop_pct: Fixed stop percentage if use_fixed_stops is True
                - max_positions: Maximum number of open positions
        """
        self.config = config
        
        # Set default parameters if not provided
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)  # 1% of equity per trade
        self.max_risk_per_day = self.config.get('max_risk_per_day', 0.05)  # 5% of equity per day
        self.max_drawdown = self.config.get('max_drawdown', 0.15)  # 15% max drawdown
        self.stop_multiplier = self.config.get('stop_multiplier', 1.5)  # 1.5 * ATR for stop distance
        self.atr_period = self.config.get('atr_period', 14)  # 14-period ATR
        self.trailing_stop = self.config.get('trailing_stop', True)  # Use trailing stops
        self.take_profit_ratio = self.config.get('take_profit_ratio', 2.0)  # 1:2 risk-reward ratio
        self.initial_equity = self.config.get('initial_equity', 1000.0)  # Initial account equity
        self.use_fixed_stops = self.config.get('use_fixed_stops', False)  # Use fixed percentage stops
        self.fixed_stop_pct = self.config.get('fixed_stop_pct', 0.03)  # 3% fixed stop
        self.max_positions = self.config.get('max_positions', 5)  # Maximum number of open positions
        
        # Internal state
        self.current_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.daily_risk_used = 0.0
        self.open_positions = []
        self.trade_history = []
        self.current_drawdown = 0.0
        self.risk_adjustment = 1.0  # Risk adjustment factor based on drawdown
        
    def calculate_position_size(self, entry_price: float, stop_price: float, df: pd.DataFrame = None) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            entry_price: Entry price
            stop_price: Stop-loss price
            df: OHLCV data (optional, used for ATR-based stops)
            
        Returns:
            Position size (quantity)
        """
        # Calculate risk amount in account currency
        risk_amount = self.current_equity * self.risk_per_trade * self.risk_adjustment
        
        # Calculate stop distance
        stop_distance_pct = abs(entry_price - stop_price) / entry_price
        
        # Calculate position size
        position_size = risk_amount / (entry_price * stop_distance_pct)
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, direction: str, df: pd.DataFrame) -> float:
        """
        Calculate stop-loss price based on ATR or fixed percentage.
        
        Args:
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            df: OHLCV data
            
        Returns:
            Stop-loss price
        """
        if self.use_fixed_stops:
            # Fixed percentage stop-loss
            if direction == 'long':
                return entry_price * (1 - self.fixed_stop_pct)
            else:  # short
                return entry_price * (1 + self.fixed_stop_pct)
        else:
            # ATR-based stop-loss
            # Calculate ATR if not already in the DataFrame
            if 'atr' not in df.columns:
                df = self._calculate_atr(df)
                
            # Get the latest ATR value
            atr = df['atr'].iloc[-1]
            
            # Calculate stop distance
            stop_distance = atr * self.stop_multiplier
            
            # Calculate stop price
            if direction == 'long':
                return entry_price - stop_distance
            else:  # short
                return entry_price + stop_distance
    
    def calculate_take_profit(self, entry_price: float, stop_price: float, direction: str) -> float:
        """
        Calculate take-profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_price: Stop-loss price
            direction: Trade direction ('long' or 'short')
            
        Returns:
            Take-profit price
        """
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_price)
        
        # Calculate take-profit distance
        tp_distance = stop_distance * self.take_profit_ratio
        
        # Calculate take-profit price
        if direction == 'long':
            return entry_price + tp_distance
        else:  # short
            return entry_price - tp_distance
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update account equity and recalculate drawdown.
        
        Args:
            new_equity: New account equity
        """
        self.current_equity = new_equity
        
        # Update peak equity if new equity is higher
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            self.current_drawdown = 0.0
        else:
            # Calculate current drawdown
            self.current_drawdown = (self.peak_equity - new_equity) / self.peak_equity
            
        # Adjust risk based on drawdown
        self._adjust_risk_for_drawdown()
        
    def _adjust_risk_for_drawdown(self) -> None:
        """
        Adjust risk allocation based on drawdown.
        
        Reduce risk as drawdown increases to protect capital.
        """
        if self.current_drawdown >= self.max_drawdown:
            # Severe drawdown, reduce risk by half
            self.risk_adjustment = 0.5
        elif self.current_drawdown >= self.max_drawdown * 0.75:
            # Significant drawdown, reduce risk by 25%
            self.risk_adjustment = 0.75
        elif self.current_drawdown >= self.max_drawdown * 0.5:
            # Moderate drawdown, reduce risk by 10%
            self.risk_adjustment = 0.9
        else:
            # Normal operation
            self.risk_adjustment = 1.0
            
    def can_open_position(self) -> bool:
        """
        Check if a new position can be opened based on risk limits.
        
        Returns:
            True if a new position can be opened, False otherwise
        """
        # Check if daily risk limit is reached
        if self.daily_risk_used >= self.max_risk_per_day:
            logger.info("Daily risk limit reached, cannot open new position")
            return False
            
        # Check if maximum number of positions is reached
        if len(self.open_positions) >= self.max_positions:
            logger.info("Maximum number of positions reached, cannot open new position")
            return False
            
        # Check if drawdown is too high
        if self.current_drawdown >= self.max_drawdown:
            logger.info("Maximum drawdown reached, cannot open new position")
            return False
            
        return True
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with ATR
        """
        df = df.copy()
        
        # Calculate true range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=self.atr_period).mean()
        
        return df
        
    def register_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Register a trade in the risk management system.
        
        Args:
            trade_data: Dictionary with trade information
        """
        # Add trade to history
        self.trade_history.append(trade_data)
        
        # Update daily risk
        self.daily_risk_used += trade_data.get('risk_pct', 0.0)
        
        # Add to open positions if the trade is open
        if trade_data.get('status') == 'open':
            self.open_positions.append(trade_data)
            
    def update_trade(self, trade_id: str, update_data: Dict[str, Any]) -> None:
        """
        Update an existing trade in the risk management system.
        
        Args:
            trade_id: Trade ID
            update_data: Dictionary with updated trade information
        """
        # Update in trade history
        for i, trade in enumerate(self.trade_history):
            if trade.get('id') == trade_id:
                self.trade_history[i].update(update_data)
                break
                
        # Update in open positions if the trade is closed
        if update_data.get('status') == 'closed':
            for i, position in enumerate(self.open_positions):
                if position.get('id') == trade_id:
                    # Remove from open positions
                    self.open_positions.pop(i)
                    break
                    
    def update_trailing_stop(self, trade_id: str, current_price: float) -> Optional[float]:
        """
        Update trailing stop for an open trade.
        
        Args:
            trade_id: Trade ID
            current_price: Current price
            
        Returns:
            New stop-loss price if updated, None otherwise
        """
        if not self.trailing_stop:
            return None
            
        # Find the trade
        trade = None
        for position in self.open_positions:
            if position.get('id') == trade_id:
                trade = position
                break
                
        if trade is None:
            return None
            
        direction = trade.get('direction')
        current_stop = trade.get('stop_price')
        entry_price = trade.get('entry_price')
        
        # Update trailing stop for long position
        if direction == 'long' and current_price > entry_price:
            # Calculate new stop level
            new_stop = current_price * (1 - self.fixed_stop_pct)
            
            # Only move stop up
            if new_stop > current_stop:
                return new_stop
                
        # Update trailing stop for short position
        elif direction == 'short' and current_price < entry_price:
            # Calculate new stop level
            new_stop = current_price * (1 + self.fixed_stop_pct)
            
            # Only move stop down
            if new_stop < current_stop:
                return new_stop
                
        return None
        
    def reset_daily_risk(self) -> None:
        """
        Reset daily risk allocation (to be called at the start of each trading day).
        """
        self.daily_risk_used = 0.0
        
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a risk report.
        
        Returns:
            Dictionary with risk information
        """
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.current_drawdown * 100,  # as percentage
            'risk_adjustment': self.risk_adjustment,
            'daily_risk_used': self.daily_risk_used * 100,  # as percentage
            'open_positions': len(self.open_positions),
            'trade_count': len(self.trade_history),
            'profitable_trades': sum(1 for t in self.trade_history if t.get('status') == 'closed' and t.get('profit', 0) > 0),
            'losing_trades': sum(1 for t in self.trade_history if t.get('status') == 'closed' and t.get('profit', 0) <= 0),
            'average_profit': np.mean([t.get('profit', 0) for t in self.trade_history if t.get('status') == 'closed']) if self.trade_history else 0,
            'largest_win': max([t.get('profit', 0) for t in self.trade_history if t.get('status') == 'closed'] or [0]),
            'largest_loss': min([t.get('profit', 0) for t in self.trade_history if t.get('status') == 'closed'] or [0])
        }