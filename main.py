import json
from datetime import datetime, timedelta
from data.fetch import fetch_historical_data
from data.save import save_to_sqlite
from data.load import load_from_sqlite
from signals.generate import generate_dummy_signals
from portfolio.calculate import calculate_portfolio_value
from plotting.plot import plot_data

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
    print(df)
    save_to_sqlite(df, symbol, timeframe, db_name)
    
    # Load data and process
    df = load_from_sqlite(symbol, timeframe, db_name)
    buy_points, sell_points = generate_dummy_signals(df, num_signals)
    portfolio = calculate_portfolio_value(df, buy_points, sell_points, initial_cash)
    
    # Plot results
    plot_data(df, buy_points, sell_points, portfolio, symbol)

if __name__ == "__main__":
    main()