import pandas as pd
import numpy as np
import ccxt
import time
import logging
from typing import Tuple, Dict, Any, List
from .base import Strategy

logger = logging.getLogger(__name__)


class TriangularArbitrage(Strategy):
    """
    Triangular Arbitrage strategy implementation.
    
    Looks for arbitrage opportunities between three trading pairs on the same exchange.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TriangularArbitrage strategy with configuration parameters.
        
        Args:
            config: Dictionary containing strategy parameters
                - min_profit_pct: Minimum profit percentage to consider an arbitrage opportunity
                - fee_pct: Trading fee percentage
                - base_asset: Base asset to start and end with (e.g., 'USDT')
                - pairs: List of trading pairs to consider for triangular arbitrage
        """
        super().__init__(config)
        
        # Set default parameters if not provided
        self.min_profit_pct = self.config.get('min_profit_pct', 0.5)  # 0.5%
        self.fee_pct = self.config.get('fee_pct', 0.1)  # 0.1% per trade
        self.base_asset = self.config.get('base_asset', 'USDT')
        self.pairs = self.config.get('pairs', [])
        self.exchange = self.config.get('exchange', 'binance')
        
        # Initialize exchange
        try:
            self.exchange_client = getattr(ccxt, self.exchange)()
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange} exchange: {e}")
            self.exchange_client = None
            
    def find_triangular_opportunities(self) -> List[Dict[str, Any]]:
        """
        Find triangular arbitrage opportunities.
        
        Returns:
            List of dictionaries containing arbitrage opportunities
        """
        if self.exchange_client is None:
            logger.error("Exchange client is not initialized")
            return []
            
        opportunities = []
        
        try:
            # Fetch ticker data for all markets
            tickers = self.exchange_client.fetch_tickers()
            
            # Generate triangular paths
            triangles = self._generate_triangles(tickers.keys())
            
            # Check each triangle for arbitrage opportunity
            for triangle in triangles:
                opportunity = self._check_triangle(triangle, tickers)
                if opportunity and opportunity['profit_pct'] > self.min_profit_pct:
                    opportunities.append(opportunity)
                    
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {e}")
            
        return opportunities
        
    def _generate_triangles(self, markets: List[str]) -> List[List[str]]:
        """
        Generate possible triangular paths based on available markets.
        
        Args:
            markets: List of available markets
            
        Returns:
            List of triangular paths
        """
        triangles = []
        
        # Filter for markets with the base asset
        base_markets = [m for m in markets if self.base_asset in m.split('/')]
        
        # For each pair of base markets, check if they can form a triangle
        for market1 in base_markets:
            for market2 in base_markets:
                if market1 == market2:
                    continue
                    
                # Extract currencies
                curr1 = market1.split('/')[0] if market1.split('/')[1] == self.base_asset else market1.split('/')[1]
                curr2 = market2.split('/')[0] if market2.split('/')[1] == self.base_asset else market2.split('/')[1]
                
                # Check if there's a market between curr1 and curr2
                if f"{curr1}/{curr2}" in markets:
                    triangles.append([market1, f"{curr1}/{curr2}", market2])
                elif f"{curr2}/{curr1}" in markets:
                    triangles.append([market1, f"{curr2}/{curr1}", market2])
                    
        return triangles
        
    def _check_triangle(self, triangle: List[str], tickers: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Check if a triangular path offers an arbitrage opportunity.
        
        Args:
            triangle: List of three markets forming a triangle
            tickers: Dictionary of ticker data
            
        Returns:
            Dictionary with arbitrage opportunity details or None
        """
        # Extract markets
        market1, market2, market3 = triangle
        
        # Extract prices
        try:
            price1 = tickers[market1]['bid'] if self.base_asset == market1.split('/')[1] else 1 / tickers[market1]['ask']
            price2 = tickers[market2]['bid'] if market2.split('/')[1] == market1.split('/')[0] else 1 / tickers[market2]['ask']
            price3 = tickers[market3]['bid'] if market3.split('/')[0] == self.base_asset else 1 / tickers[market3]['ask']
            
            # Calculate cross rate
            cross_rate = price1 * price2 * price3
            
            # Calculate profit percentage (accounting for fees)
            fee_factor = (1 - self.fee_pct / 100) ** 3  # Three trades
            profit_pct = (cross_rate * fee_factor - 1) * 100
            
            if profit_pct > 0:
                return {
                    'triangle': triangle,
                    'cross_rate': cross_rate,
                    'profit_pct': profit_pct,
                    'timestamp': pd.Timestamp.now()
                }
                
        except Exception as e:
            logger.error(f"Error checking triangle {triangle}: {e}")
            
        return None
    
    def generate_signals(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate buy and sell signals based on arbitrage opportunities.
        Note: For arbitrage, this would typically be called without historical data.
        This method is implemented for compatibility with the Strategy interface.
        
        Args:
            df: OHLCV data (not used directly for arbitrage)
            
        Returns:
            Tuple of (buy_signals, sell_signals) DataFrames
        """
        # This is a placeholder as arbitrage doesn't use historical OHLCV data in the same way
        # In a real implementation, you would have a separate method that runs continuously
        
        # For testing/simulation purposes, we can create some dummy signals
        if df.empty:
            return pd.DataFrame(columns=['timestamp', 'price']), pd.DataFrame(columns=['timestamp', 'price'])
            
        # We'll create a signal at the current timestamp
        current_time = pd.Timestamp.now()
        opportunities = self.find_triangular_opportunities()
        
        if opportunities:
            # If we found opportunities, create a buy signal
            buy_signal = pd.DataFrame({
                'timestamp': [current_time],
                'price': [df['close'].iloc[-1] if not df.empty else 0],
                'opportunity': [opportunities[0]]  # Include the best opportunity
            })
            
            # For arbitrage, sell signal would follow immediately
            sell_signal = pd.DataFrame({
                'timestamp': [current_time + pd.Timedelta(minutes=1)],
                'price': [df['close'].iloc[-1] if not df.empty else 0],
                'opportunity': [opportunities[0]]
            })
            
            return buy_signal[['timestamp', 'price']], sell_signal[['timestamp', 'price']]
        
        return pd.DataFrame(columns=['timestamp', 'price']), pd.DataFrame(columns=['timestamp', 'price'])
    
    def validate_parameters(self) -> bool:
        """
        Validate strategy parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        if self.min_profit_pct <= 0:
            print("Error: min_profit_pct must be positive")
            return False
            
        if self.fee_pct < 0:
            print("Error: fee_pct cannot be negative")
            return False
            
        if not self.base_asset:
            print("Error: base_asset must be specified")
            return False
            
        if self.exchange_client is None:
            print(f"Error: Could not initialize {self.exchange} exchange")
            return False
            
        return True
        
    def execute_arbitrage(self, opportunity: Dict[str, Any], amount: float) -> Dict[str, Any]:
        """
        Execute an arbitrage opportunity.
        
        Args:
            opportunity: Dictionary with arbitrage opportunity details
            amount: Amount of base asset to use
            
        Returns:
            Dictionary with execution results
        """
        if self.exchange_client is None:
            logger.error("Exchange client is not initialized")
            return {'success': False, 'error': 'Exchange client not initialized'}
            
        triangle = opportunity['triangle']
        
        try:
            # Execute first trade
            market1 = triangle[0]
            side1 = 'buy' if self.base_asset == market1.split('/')[1] else 'sell'
            symbol1 = market1
            
            if side1 == 'buy':
                amount1 = amount  # Amount of base asset to spend
                price1 = None  # Market order
                response1 = self.exchange_client.create_order(symbol1, 'market', side1, amount1)
                amount2 = response1['amount']  # Amount of asset received
            else:
                amount1 = amount  # Amount of asset to sell
                price1 = None  # Market order
                response1 = self.exchange_client.create_order(symbol1, 'market', side1, amount1)
                amount2 = response1['cost']  # Amount of base asset received
                
            # Execute second trade
            market2 = triangle[1]
            symbol2 = market2
            side2 = 'buy' if market1.split('/')[0] == market2.split('/')[1] else 'sell'
            
            if side2 == 'buy':
                amount2_adj = amount2  # Amount to spend
                price2 = None  # Market order
                response2 = self.exchange_client.create_order(symbol2, 'market', side2, amount2_adj)
                amount3 = response2['amount']  # Amount received
            else:
                amount2_adj = amount2  # Amount to sell
                price2 = None  # Market order
                response2 = self.exchange_client.create_order(symbol2, 'market', side2, amount2_adj)
                amount3 = response2['cost']  # Amount received
                
            # Execute third trade
            market3 = triangle[2]
            symbol3 = market3
            side3 = 'buy' if self.base_asset == market3.split('/')[1] else 'sell'
            
            if side3 == 'buy':
                amount3_adj = amount3  # Amount to spend
                price3 = None  # Market order
                response3 = self.exchange_client.create_order(symbol3, 'market', side3, amount3_adj)
                final_amount = response3['amount']  # Final amount of base asset
            else:
                amount3_adj = amount3  # Amount to sell
                price3 = None  # Market order
                response3 = self.exchange_client.create_order(symbol3, 'market', side3, amount3_adj)
                final_amount = response3['cost']  # Final amount of base asset
                
            # Calculate actual profit
            profit = final_amount - amount
            profit_pct = (profit / amount) * 100
            
            return {
                'success': True,
                'triangle': triangle,
                'initial_amount': amount,
                'final_amount': final_amount,
                'profit': profit,
                'profit_pct': profit_pct,
                'trades': [response1, response2, response3]
            }
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {e}")
            return {'success': False, 'error': str(e)}