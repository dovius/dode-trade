import json
from datetime import datetime, timedelta
from data.fetch import fetch_historical_data
from data.save import save_to_sqlite
from data.load import load_from_sqlite
from plotting.plot_summary import plot_summary, compare_algorithms
from signals.generate_moving_avg_crossover_signals import generate_moving_avg_crossover_signals
from signals.generate_london_breakout_signals import generate_london_breakout_signals
from signals.generate_heikin_ashi_signals import generate_heikin_ashi_signals
from signals.generate_rsi_signals import generate_rsi_signals
from signals.generate_bollinger_bands_signals import generate_bollinger_bands_signals
from portfolio.calculate import calculate_portfolio_value
from plotting.plot import plot_data

trading_algorithms = [
    {'name': 'Moving Average Crossover', 'trading_fn': generate_moving_avg_crossover_signals},
    {'name': 'London Breakout', 'trading_fn': generate_london_breakout_signals},
    {'name': 'Heikin-Ashi', 'trading_fn': generate_heikin_ashi_signals},
    {'name': 'RSI', 'trading_fn': generate_rsi_signals},
    {'name': 'Bollinger Bands', 'trading_fn': generate_bollinger_bands_signals},
]

def main():
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    symbol = config['symbol']
    timeframe = config['timeframe']
    db_name = config['db_name']
    initial_cash = config['initial_cash']
    num_signals = config['num_signals']
    days = config['days']
    
    # Fetch and save data
    start_date = datetime.now() - timedelta(days=days)
    end_date = datetime.now()
    df = fetch_historical_data(symbol, timeframe, start_date, end_date)
    # save_to_sqlite(df, symbol, timeframe, db_name)
    
    # # Load data and process
    # df = load_from_sqlite(symbol, timeframe, db_name)
    buy_points, sell_points = generate_moving_avg_crossover_signals(df, num_signals)
    portfolio, trades = calculate_portfolio_value(df, buy_points, sell_points, initial_cash)
    
    # Plot results
    # plot_data(df, buy_points, sell_points, portfolio, symbol, trades)


    dashboard = compare_algorithms(df, trading_algorithms, symbol, initial_cash, num_signals)
    
    html_file = f"{symbol.replace('/', '_')}_dashboard.html"
    dashboard.write_html(html_file, auto_open=True)
    print(f"Dashboard saved to {html_file} and opened in browser")
    
if __name__ == "__main__":
    main()