import json
import logging
import os
import time
from datetime import datetime, timedelta
import argparse
import pandas as pd

# Data modules
from data.fetch import fetch_historical_data
from data.save import save_to_sqlite
from data.load import load_from_sqlite

# Strategy modules
from strategies.manager import StrategyManager
from strategies.trend_following import TrendFollowing
from strategies.mean_reversion import MeanReversion
from strategies.arbitrage import TriangularArbitrage

# Risk management
from risk.manager import RiskManager

# Trading modules
from trading.order_manager import OrderManager

# Visualization
from plotting.plot import plot_data


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_config(config_file='config/config.json'):
    """
    Load configuration from file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Dictionary with configuration
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def setup_strategies(config):
    """
    Set up trading strategies.
    
    Args:
        config: Dictionary with configuration
        
    Returns:
        StrategyManager instance
    """
    # Create strategy configurations
    strategies_config = {
        'strategies': [
            {
                'name': 'TrendFollowing',
                'module': 'strategies.trend_following',
                'class': 'TrendFollowing',
                'weight': 1.0,
                'parameters': {
                    'short_window': config.get('tf_short_window', 9),
                    'long_window': config.get('tf_long_window', 21),
                    'adx_period': config.get('tf_adx_period', 14),
                    'adx_threshold': config.get('tf_adx_threshold', 25),
                    'use_adx_filter': config.get('tf_use_adx_filter', True)
                }
            },
            {
                'name': 'MeanReversion',
                'module': 'strategies.mean_reversion',
                'class': 'MeanReversion',
                'weight': 0.8,
                'parameters': {
                    'rsi_period': config.get('mr_rsi_period', 14),
                    'rsi_oversold': config.get('mr_rsi_oversold', 30),
                    'rsi_overbought': config.get('mr_rsi_overbought', 70),
                    'bb_period': config.get('mr_bb_period', 20),
                    'bb_std': config.get('mr_bb_std', 2.0),
                    'use_rsi': config.get('mr_use_rsi', True),
                    'use_bb': config.get('mr_use_bb', True),
                    'confirmation_candles': config.get('mr_confirmation_candles', 1)
                }
            },
            {
                'name': 'TriangularArbitrage',
                'module': 'strategies.arbitrage',
                'class': 'TriangularArbitrage',
                'weight': 0.5,
                'parameters': {
                    'min_profit_pct': config.get('arb_min_profit_pct', 0.5),
                    'fee_pct': config.get('arb_fee_pct', 0.1),
                    'base_asset': config.get('arb_base_asset', 'USDT'),
                    'exchange': config.get('exchange', 'binance')
                }
            }
        ],
        'enable_aggregation': config.get('enable_strategy_aggregation', False),
        'aggregation_method': config.get('strategy_aggregation_method', 'weighted')
    }
    
    # Create strategy manager
    strategy_manager = StrategyManager(strategies_config)
    
    return strategy_manager


def setup_risk_manager(config):
    """
    Set up risk management.
    
    Args:
        config: Dictionary with configuration
        
    Returns:
        RiskManager instance
    """
    risk_config = {
        'risk_per_trade': config.get('risk_per_trade', 0.01),
        'max_risk_per_day': config.get('max_risk_per_day', 0.05),
        'max_drawdown': config.get('max_drawdown', 0.15),
        'stop_multiplier': config.get('stop_multiplier', 1.5),
        'atr_period': config.get('atr_period', 14),
        'trailing_stop': config.get('trailing_stop', True),
        'take_profit_ratio': config.get('take_profit_ratio', 2.0),
        'initial_equity': config.get('initial_cash', 1000.0),
        'use_fixed_stops': config.get('use_fixed_stops', False),
        'fixed_stop_pct': config.get('fixed_stop_pct', 0.03),
        'max_positions': config.get('max_positions', 5)
    }
    
    risk_manager = RiskManager(risk_config)
    
    return risk_manager


def setup_order_manager(config):
    """
    Set up order management.
    
    Args:
        config: Dictionary with configuration
        
    Returns:
        OrderManager instance
    """
    order_config = {
        'exchange': config.get('exchange', 'binance'),
        'api_key': config.get('api_key', ''),
        'api_secret': config.get('api_secret', ''),
        'paper_trading': config.get('paper_trading', True),
        'symbol': config.get('symbol', 'BTC/USDT'),
        'default_type': config.get('order_type', 'market'),
        'limit_slippage': config.get('limit_slippage', 0.001)
    }
    
    order_manager = OrderManager(order_config)
    
    return order_manager


def backtest_mode(config, strategy_manager, risk_manager):
    """
    Run the trading bot in backtest mode.
    
    Args:
        config: Dictionary with configuration
        strategy_manager: StrategyManager instance
        risk_manager: RiskManager instance
    """
    logger.info("Starting backtest mode")
    
    # Load or fetch historical data
    symbol = config['symbol']
    timeframe = config['timeframe']
    db_name = config['db_name']
    days = config['days']
    
    try:
        # Load from database if available
        df = load_from_sqlite(symbol, timeframe, db_name)
        
        # If data is not available or outdated, fetch it
        if df.empty or (datetime.now() - pd.to_datetime(df['timestamp'].max())).days > 1:  # Convert timestamp to datetime
            logger.info("Fetching historical data")
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
            df = fetch_historical_data(symbol, timeframe, start_date, end_date)
            save_to_sqlite(df, symbol, timeframe, db_name)
        
        # Generate trading signals using strategy manager
        buy_points, sell_points = strategy_manager.get_signals(df)
        
        # Log signals
        logger.info(f"Generated {len(buy_points)} buy signals and {len(sell_points)} sell signals")
        
        # Calculate portfolio performance
        from portfolio.calculate import calculate_portfolio_value
        portfolio, trades = calculate_portfolio_value(df, buy_points, sell_points, config['initial_cash'])
        
        # Plot results
        plot_data(df, buy_points, sell_points, portfolio, symbol, trades)
        
        # Display performance metrics
        if not trades.empty:
            profit_trades = trades[trades['profit_percentage'] > 0]
            loss_trades = trades[trades['profit_percentage'] <= 0]
            
            total_profit = trades['profit_loss'].sum()
            win_rate = len(profit_trades) / len(trades) * 100 if len(trades) > 0 else 0
            avg_profit = profit_trades['profit_percentage'].mean() if len(profit_trades) > 0 else 0
            avg_loss = loss_trades['profit_percentage'].mean() if len(loss_trades) > 0 else 0
            
            logger.info(f"Backtest Results for {symbol}:")
            logger.info(f"Total Trades: {len(trades)}")
            logger.info(f"Win Rate: {win_rate:.2f}%")
            logger.info(f"Total Profit: {total_profit:.2f}")
            logger.info(f"Average Profit: {avg_profit:.2f}%")
            logger.info(f"Average Loss: {avg_loss:.2f}%")
            
            # Save results to file
            backtest_results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': df['timestamp'].min().strftime('%Y-%m-%d'),
                'end_date': df['timestamp'].max().strftime('%Y-%m-%d'),
                'total_trades': len(trades),
                'win_rate': float(f"{win_rate:.2f}"),
                'total_profit': float(f"{total_profit:.2f}"),
                'avg_profit': float(f"{avg_profit:.2f}"),
                'avg_loss': float(f"{avg_loss:.2f}"),
                'initial_cash': config['initial_cash'],
                'final_value': float(f"{portfolio['total_value'].iloc[-1]:.2f}")
            }
            
            # Save backtest results
            if not os.path.exists('results'):
                os.makedirs('results')
                
            with open(f"results/backtest_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                json.dump(backtest_results, f, indent=2)
        else:
            logger.warning("No trades were executed during the backtest period")
    
    except Exception as e:
        logger.error(f"Error in backtest mode: {e}")


def live_mode(config, strategy_manager, risk_manager, order_manager):
    """
    Run the trading bot in live mode.
    
    Args:
        config: Dictionary with configuration
        strategy_manager: StrategyManager instance
        risk_manager: RiskManager instance
        order_manager: OrderManager instance
    """
    logger.info("Starting live trading mode")
    
    symbol = config['symbol']
    timeframe = config['timeframe']
    
    try:
        while True:
            # Fetch latest data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config.get('data_window', 30))
            
            logger.info(f"Fetching latest data for {symbol}")
            df = fetch_historical_data(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                logger.error("Failed to fetch data, retrying in 1 minute")
                time.sleep(60)
                continue
                
            # Generate trading signals
            buy_points, sell_points = strategy_manager.get_signals(df)
            
            # Check if we have any signals to act on
            latest_candle_time = df['timestamp'].iloc[-1]
            latest_price = df['close'].iloc[-1]
            
            # Process buy signals
            for _, signal in buy_points.iterrows():
                if (datetime.now() - signal['timestamp']).total_seconds() < config.get('signal_expiry', 300):
                    # Check if we can open a new position
                    if risk_manager.can_open_position():
                        # Calculate position size and stop loss
                        stop_price = risk_manager.calculate_stop_loss(latest_price, 'long', df)
                        position_size = risk_manager.calculate_position_size(latest_price, stop_price, df)
                        
                        # Calculate take profit level
                        take_profit = risk_manager.calculate_take_profit(latest_price, stop_price, 'long')
                        
                        # Place order
                        order_result = order_manager.place_order({
                            'symbol': symbol,
                            'side': 'buy',
                            'type': 'market',
                            'amount': position_size,
                            'price': latest_price,
                            'stop_price': stop_price,
                            'take_profit': take_profit
                        })
                        
                        if order_result['success']:
                            order_id = order_result['order_id']
                            logger.info(f"Buy order placed: {order_id} for {position_size} {symbol} at {latest_price}")
                            
                            # Register the trade with risk manager
                            risk_manager.register_trade({
                                'id': order_id,
                                'symbol': symbol,
                                'direction': 'long',
                                'entry_price': latest_price,
                                'stop_price': stop_price,
                                'take_profit': take_profit,
                                'amount': position_size,
                                'time': datetime.now(),
                                'status': 'open',
                                'risk_pct': (latest_price - stop_price) / latest_price
                            })
                        else:
                            logger.error(f"Failed to place buy order: {order_result.get('error')}")
            
            # Process sell signals (for existing positions)
            for _, signal in sell_signals.iterrows():
                if (datetime.now() - signal['timestamp']).total_seconds() < config.get('signal_expiry', 300):
                    # Get open positions
                    open_orders = order_manager.get_open_orders(symbol)
                    
                    # Close all open positions that match the symbol
                    for order in open_orders:
                        if order['side'] == 'buy' and order['status'] == 'closed':
                            # Place sell order to close the position
                            close_result = order_manager.place_order({
                                'symbol': symbol,
                                'side': 'sell',
                                'type': 'market',
                                'amount': order['amount'],
                                'price': latest_price
                            })
                            
                            if close_result['success']:
                                order_id = close_result['order_id']
                                logger.info(f"Sell order placed: {order_id} for {order['amount']} {symbol} at {latest_price}")
                                
                                # Update the trade in risk manager
                                profit_loss = (latest_price - order['price']) * order['amount']
                                risk_manager.update_trade(order['id'], {
                                    'status': 'closed',
                                    'exit_price': latest_price,
                                    'exit_time': datetime.now(),
                                    'profit_loss': profit_loss,
                                    'profit_pct': (latest_price - order['price']) / order['price'] * 100,
                                    'is_profitable': profit_loss > 0
                                })
                            else:
                                logger.error(f"Failed to place sell order: {close_result.get('error')}")
            
            # Update risk manager with current equity
            # In live trading, you would get this from the exchange
            # For paper trading, calculate based on orders
            
            # Wait for next update
            sleep_time = config.get('update_interval', 60)  # Default to 1 minute
            logger.info(f"Sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error in live trading mode: {e}")


def main():
    """
    Main function to run the trading bot.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest', help='Trading mode')
    parser.add_argument('--config', default='config/config.json', help='Configuration file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if not config:
        logger.error("Failed to load configuration, exiting")
        return
    
    # Set up components
    strategy_manager = setup_strategies(config)
    risk_manager = setup_risk_manager(config)
    order_manager = setup_order_manager(config)
    
    # Run in selected mode
    if args.mode == 'backtest':
        backtest_mode(config, strategy_manager, risk_manager)
    else:  # live mode
        live_mode(config, strategy_manager, risk_manager, order_manager)


if __name__ == "__main__":
    main()