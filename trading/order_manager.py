import pandas as pd
import ccxt
import logging
import uuid
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages order execution and tracking.
    
    Handles order placement, status tracking, and execution reporting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OrderManager with configuration parameters.
        
        Args:
            config: Dictionary containing order manager configuration
                - exchange: Exchange name (e.g., 'binance')
                - api_key: API key for the exchange
                - api_secret: API secret for the exchange
                - paper_trading: Whether to use paper trading
                - symbol: Trading pair symbol (e.g., 'BTC/USDT')
                - default_type: Default order type ('limit' or 'market')
                - limit_slippage: Slippage percentage for limit orders
        """
        self.config = config
        
        # Set default parameters
        self.exchange_name = self.config.get('exchange', 'binance')
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.paper_trading = self.config.get('paper_trading', True)
        self.symbol = self.config.get('symbol', 'BTC/USDT')
        self.default_type = self.config.get('default_type', 'market')
        self.limit_slippage = self.config.get('limit_slippage', 0.001)  # 0.1%
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Initialize order tracking
        self.open_orders = {}
        self.order_history = {}
        
        # Load saved orders if paper trading
        if self.paper_trading:
            self._load_paper_orders()
            
    def _initialize_exchange(self) -> ccxt.Exchange:
        """
        Initialize exchange connection.
        
        Returns:
            Exchange object
        """
        try:
            if self.paper_trading:
                # In paper trading mode, we don't need API credentials
                exchange_class = getattr(ccxt, self.exchange_name)
                exchange = exchange_class({
                    'enableRateLimit': True,
                })
                logger.info(f"Initialized paper trading on {self.exchange_name}")
            else:
                # In live trading mode, use API credentials
                exchange_class = getattr(ccxt, self.exchange_name)
                exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                })
                logger.info(f"Initialized live trading on {self.exchange_name}")
                
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            # Return a placeholder exchange for paper trading
            return ccxt.Exchange({
                'id': 'paper_trading',
                'enableRateLimit': True,
            })
            
    def _load_paper_orders(self) -> None:
        """
        Load paper trading orders from file.
        """
        try:
            with open('paper_orders.json', 'r') as f:
                data = json.load(f)
                self.open_orders = data.get('open_orders', {})
                self.order_history = data.get('order_history', {})
        except FileNotFoundError:
            logger.info("No paper orders file found. Starting with empty order history.")
        except Exception as e:
            logger.error(f"Error loading paper orders: {e}")
            
    def _save_paper_orders(self) -> None:
        """
        Save paper trading orders to file.
        """
        if not self.paper_trading:
            return
            
        try:
            with open('paper_orders.json', 'w') as f:
                json.dump({
                    'open_orders': self.open_orders,
                    'order_history': self.order_history
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving paper orders: {e}")
            
    def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Place an order on the exchange.
        
        Args:
            order_data: Dictionary containing order information
                - symbol: Trading pair symbol (e.g., 'BTC/USDT')
                - side: Order side ('buy' or 'sell')
                - type: Order type ('limit' or 'market')
                - amount: Order amount
                - price: Order price (for limit orders)
                - stop_price: Stop price (for stop orders)
                - take_profit: Take profit price
                - client_order_id: Client order ID
                - leverage: Leverage value (for futures trading)
                
        Returns:
            Dictionary with order result
        """
        # Extract order parameters
        symbol = order_data.get('symbol', self.symbol)
        side = order_data.get('side', 'buy')
        order_type = order_data.get('type', self.default_type)
        amount = order_data.get('amount', 0.0)
        price = order_data.get('price')
        stop_price = order_data.get('stop_price')
        take_profit = order_data.get('take_profit')
        client_order_id = order_data.get('client_order_id', str(uuid.uuid4()))
        leverage = order_data.get('leverage')
        
        # Validate order parameters
        if amount <= 0:
            return {
                'success': False,
                'error': 'Invalid order amount'
            }
            
        if order_type == 'limit' and price is None:
            return {
                'success': False,
                'error': 'Price is required for limit orders'
            }
            
        if self.paper_trading:
            # Execute paper trade
            return self._place_paper_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                stop_price=stop_price,
                take_profit=take_profit,
                client_order_id=client_order_id
            )
        else:
            # Execute real trade
            try:
                # Set leverage if provided (for futures trading)
                if leverage is not None:
                    try:
                        self.exchange.set_leverage(leverage, symbol)
                    except Exception as e:
                        logger.warning(f"Failed to set leverage: {e}")
                
                # Prepare order parameters
                params = {}
                
                if stop_price is not None:
                    params['stopPrice'] = stop_price
                    
                if take_profit is not None:
                    params['takeProfit'] = take_profit
                    
                # Place the order
                response = self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price if order_type == 'limit' else None,
                    params={'clientOrderId': client_order_id, **params}
                )
                
                # Process response
                order_id = response.get('id')
                if order_id:
                    self.open_orders[order_id] = {
                        'id': order_id,
                        'client_order_id': client_order_id,
                        'symbol': symbol,
                        'side': side,
                        'type': order_type,
                        'amount': amount,
                        'price': price,
                        'stop_price': stop_price,
                        'take_profit': take_profit,
                        'status': 'open',
                        'timestamp': datetime.now().timestamp(),
                        'response': response
                    }
                    
                return {
                    'success': True,
                    'order_id': order_id,
                    'client_order_id': client_order_id,
                    'response': response
                }
                
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
                
    def _place_paper_order(self, symbol: str, side: str, order_type: str, amount: float, 
                          price: Optional[float] = None, stop_price: Optional[float] = None, 
                          take_profit: Optional[float] = None, client_order_id: str = "") -> Dict[str, Any]:
        """
        Place a paper trading order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type ('limit' or 'market')
            amount: Order amount
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            take_profit: Take profit price
            client_order_id: Client order ID
            
        Returns:
            Dictionary with order result
        """
        # Generate a unique order ID
        order_id = str(uuid.uuid4())
        
        # Get current market price
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            current_price = price or 0
            
        # For market orders, use current price
        if order_type == 'market':
            execution_price = current_price
        else:
            # For limit orders, use the specified price
            execution_price = price
            
            # Check if the limit order would be filled immediately
            if (side == 'buy' and execution_price >= current_price) or \
               (side == 'sell' and execution_price <= current_price):
                # Immediate fill
                status = 'closed'
            else:
                # Pending fill
                status = 'open'
        
        # Create paper order
        order = {
            'id': order_id,
            'client_order_id': client_order_id,
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'amount': amount,
            'price': execution_price,
            'stop_price': stop_price,
            'take_profit': take_profit,
            'status': 'closed' if order_type == 'market' else status,
            'timestamp': datetime.now().timestamp(),
            'filled': amount if order_type == 'market' or status == 'closed' else 0.0,
            'cost': amount * execution_price if order_type == 'market' or status == 'closed' else 0.0,
            'fee': amount * execution_price * 0.001 if order_type == 'market' or status == 'closed' else 0.0  # Assuming 0.1% fee
        }
        
        # Store the order
        if order['status'] == 'open':
            self.open_orders[order_id] = order
        else:
            self.order_history[order_id] = order
            
        # Save paper orders
        self._save_paper_orders()
        
        return {
            'success': True,
            'order_id': order_id,
            'client_order_id': client_order_id,
            'response': order
        }
        
    def cancel_order(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with cancellation result
        """
        if self.paper_trading:
            # Cancel paper order
            if order_id in self.open_orders:
                order = self.open_orders.pop(order_id)
                order['status'] = 'canceled'
                self.order_history[order_id] = order
                
                # Save paper orders
                self._save_paper_orders()
                
                return {
                    'success': True,
                    'order_id': order_id,
                    'response': order
                }
            else:
                return {
                    'success': False,
                    'error': f"Order {order_id} not found"
                }
        else:
            # Cancel real order
            try:
                if symbol is None:
                    # Try to get symbol from open orders
                    if order_id in self.open_orders:
                        symbol = self.open_orders[order_id]['symbol']
                    else:
                        return {
                            'success': False,
                            'error': f"Symbol required to cancel order {order_id}"
                        }
                        
                response = self.exchange.cancel_order(order_id, symbol)
                
                # Update order status
                if order_id in self.open_orders:
                    order = self.open_orders.pop(order_id)
                    order['status'] = 'canceled'
                    self.order_history[order_id] = order
                    
                return {
                    'success': True,
                    'order_id': order_id,
                    'response': response
                }
            except Exception as e:
                logger.error(f"Order cancellation failed: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
                
    def get_order_status(self, order_id: str, symbol: str = None) -> Dict[str, Any]:
        """
        Get status of an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dictionary with order status
        """
        if self.paper_trading:
            # Get paper order status
            if order_id in self.open_orders:
                return {
                    'success': True,
                    'order': self.open_orders[order_id]
                }
            elif order_id in self.order_history:
                return {
                    'success': True,
                    'order': self.order_history[order_id]
                }
            else:
                return {
                    'success': False,
                    'error': f"Order {order_id} not found"
                }
        else:
            # Get real order status
            try:
                if symbol is None:
                    # Try to get symbol from open orders or order history
                    if order_id in self.open_orders:
                        symbol = self.open_orders[order_id]['symbol']
                    elif order_id in self.order_history:
                        symbol = self.order_history[order_id]['symbol']
                    else:
                        return {
                            'success': False,
                            'error': f"Symbol required to get order {order_id}"
                        }
                        
                order = self.exchange.fetch_order(order_id, symbol)
                
                # Update order status
                status = order.get('status')
                if status == 'closed' or status == 'canceled':
                    if order_id in self.open_orders:
                        self.order_history[order_id] = self.open_orders.pop(order_id)
                        self.order_history[order_id]['status'] = status
                        
                return {
                    'success': True,
                    'order': order
                }
            except Exception as e:
                logger.error(f"Failed to get order status: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
                
    def update_paper_orders(self, current_price: float, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Update paper trading orders based on current price.
        
        Args:
            current_price: Current market price
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of updated orders
        """
        if not self.paper_trading:
            return []
            
        updated_orders = []
        
        for order_id, order in list(self.open_orders.items()):
            if symbol is not None and order['symbol'] != symbol:
                continue
                
            # Check if limit orders should be filled
            if order['type'] == 'limit':
                if order['side'] == 'buy' and current_price <= order['price']:
                    # Fill buy limit order
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['price']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                elif order['side'] == 'sell' and current_price >= order['price']:
                    # Fill sell limit order
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['price']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                    
            # Check if stop orders should be triggered
            if order['stop_price'] is not None:
                if order['side'] == 'sell' and current_price <= order['stop_price']:
                    # Trigger sell stop order
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['stop_price']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                elif order['side'] == 'buy' and current_price >= order['stop_price']:
                    # Trigger buy stop order
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['stop_price']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                    
            # Check if take profit orders should be triggered
            if order['take_profit'] is not None:
                if order['side'] == 'sell' and current_price >= order['take_profit']:
                    # Trigger sell take profit
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['take_profit']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                elif order['side'] == 'buy' and current_price <= order['take_profit']:
                    # Trigger buy take profit
                    order['status'] = 'closed'
                    order['filled'] = order['amount']
                    order['cost'] = order['amount'] * order['take_profit']
                    order['fee'] = order['cost'] * 0.001  # Assuming 0.1% fee
                    self.order_history[order_id] = order
                    del self.open_orders[order_id]
                    updated_orders.append(order)
                    
        if updated_orders:
            # Save paper orders
            self._save_paper_orders()
            
        return updated_orders
        
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of open orders
        """
        if self.paper_trading:
            # Get paper open orders
            if symbol is None:
                return list(self.open_orders.values())
            else:
                return [order for order in self.open_orders.values() if order['symbol'] == symbol]
        else:
            # Get real open orders
            try:
                orders = self.exchange.fetch_open_orders(symbol)
                
                # Update local tracking
                for order in orders:
                    order_id = order['id']
                    self.open_orders[order_id] = {
                        'id': order_id,
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'type': order['type'],
                        'amount': order['amount'],
                        'price': order['price'],
                        'status': order['status'],
                        'timestamp': order['timestamp'] / 1000,  # Convert from milliseconds
                        'response': order
                    }
                    
                return orders
            except Exception as e:
                logger.error(f"Failed to fetch open orders: {e}")
                return []
                
    def get_order_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history.
        
        Args:
            symbol: Trading pair symbol (optional)
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        if self.paper_trading:
            # Get paper order history
            history = list(self.order_history.values())
            
            if symbol is not None:
                history = [order for order in history if order['symbol'] == symbol]
                
            # Sort by timestamp (most recent first)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return history[:limit]
        else:
            # Get real order history
            try:
                orders = self.exchange.fetch_closed_orders(symbol, limit=limit)
                
                # Update local tracking
                for order in orders:
                    order_id = order['id']
                    self.order_history[order_id] = {
                        'id': order_id,
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'type': order['type'],
                        'amount': order['amount'],
                        'price': order['price'],
                        'status': order['status'],
                        'timestamp': order['timestamp'] / 1000,  # Convert from milliseconds
                        'response': order
                    }
                    
                return orders
            except Exception as e:
                logger.error(f"Failed to fetch order history: {e}")
                return []
                
    def create_oco_order(self, symbol: str, side: str, amount: float, 
                         price: float, stop_price: float, stop_limit_price: float = None) -> Dict[str, Any]:
        """
        Create an OCO (One-Cancels-Other) order.
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Limit price
            stop_price: Stop price
            stop_limit_price: Stop limit price (optional)
            
        Returns:
            Dictionary with order result
        """
        if self.paper_trading:
            # Create paper OCO order
            # This is simulated as two separate orders (limit and stop)
            limit_order = self._place_paper_order(
                symbol=symbol,
                side=side,
                order_type='limit',
                amount=amount,
                price=price
            )
            
            stop_order = self._place_paper_order(
                symbol=symbol,
                side=side,
                order_type='stop',
                amount=amount,
                price=stop_limit_price or stop_price,
                stop_price=stop_price
            )
            
            # Link the orders as OCO
            if limit_order['success'] and stop_order['success']:
                limit_order_id = limit_order['order_id']
                stop_order_id = stop_order['order_id']
                
                # Mark both orders as part of OCO
                if limit_order_id in self.open_orders:
                    self.open_orders[limit_order_id]['oco'] = stop_order_id
                    
                if stop_order_id in self.open_orders:
                    self.open_orders[stop_order_id]['oco'] = limit_order_id
                    
                # Save paper orders
                self._save_paper_orders()
                
                return {
                    'success': True,
                    'limit_order_id': limit_order_id,
                    'stop_order_id': stop_order_id
                }
            else:
                # Cancel any created order
                if limit_order['success']:
                    self.cancel_order(limit_order['order_id'])
                    
                if stop_order['success']:
                    self.cancel_order(stop_order['order_id'])
                    
                return {
                    'success': False,
                    'error': 'Failed to create OCO order'
                }
        else:
            # Create real OCO order
            try:
                response = self.exchange.create_order_oco(
                    symbol=symbol,
                    side=side,
                    amount=amount,
                    price=price,
                    stopPrice=stop_price,
                    stopLimitPrice=stop_limit_price
                )
                
                return {
                    'success': True,
                    'response': response
                }
            except Exception as e:
                logger.error(f"Failed to create OCO order: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }