# main.py

import os
from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder
from predictor import PricePredictor
from data_simulator import DataSimulator
from data_manager import DataManager
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main(use_simulation=True):
    # Initialize components
    api_key = os.getenv("API_KEY") if not use_simulation else None
    api_secret = os.getenv("API_SECRET") if not use_simulation else None
    
    # Create instances
    collector = DataCollector(api_key, api_secret, use_simulation=use_simulation)
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    # Collect and prepare data
    df = collector.get_historical_data(
        symbol="BTCUSDT",
        interval="1h",
        start_str="1 month ago UTC"
    )
    
    # Rest of the code remains the same as in the previous version
    df = engineer.add_technical_indicators(df)
    X, y = preprocessor.prepare_data(df)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = ModelBuilder.build_lstm_model(
        sequence_length=60,
        n_features=9
    )
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    predictor = PricePredictor(model, preprocessor)
    last_sequence = X_test[-1:]
    next_price = predictor.predict_next_price(last_sequence)
    print(f"Predicted next price: ${next_price:.2f}")


# Example usage script
def download_and_store_historical_data():
    """Script to download and store historical data from Binance"""
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    
    collector = DataCollector(api_key, api_secret)#use_stored_data=True)
    data_manager = DataManager()
    
    # Define what data you want to download
    symbols = ['BTCUSDT', 'ETHUSDT']
    intervals = ['1h', '4h', '1d']
    
    for symbol in symbols:
        for interval in intervals:
            print(f"Downloading {symbol} {interval} data...")
            df = collector.get_historical_data(
                symbol=symbol,
                interval=interval,
                start_str="6 months ago UTC"
            )
            
            if df is not None:
                filepath = data_manager.save_raw_data(df, symbol, interval)
                print(f"Saved to {filepath}")

def simulate_with_real_data():
    """Example of using stored data for simulation"""
    data_manager = DataManager()
    
    # Load historical data
    df = data_manager.load_data("BTCUSDT", "1h")
    
    if df is not None:
        # Create simulator with parameters based on real data
        simulator = DataSimulator(
            starting_price=df['close'].iloc[-1],
            volatility=df['close'].pct_change().std(),
            trend=df['close'].pct_change().mean()
        )
        
        # Generate simulated data with characteristics of real data
        simulated_df = simulator.generate_historical_data()
        return simulated_df
    
    return None

def view_stored_data(symbol='BTCUSDT', interval='1h', rows=5, include_stats=True):
    """
    View stored cryptocurrency data with optional statistics.
    
    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval (e.g., '1h', '4h', '1d')
        rows (int): Number of rows to display from start and end
        include_stats (bool): Whether to show basic statistics
    """
    data_manager = DataManager()
    df = data_manager.load_data(symbol, interval)
    
    if df is None:
        print(f"No data found for {symbol} at {interval} interval")
        return
        
    print(f"\n=== {symbol} {interval} Data ===")
    print("\nFirst {rows} rows:")
    print(df.head(rows))
    print(f"\nLast {rows} rows:")
    print(df.tail(rows))
    
    if include_stats:
        print("\nBasic Statistics:")
        print(df[['open', 'high', 'low', 'close', 'volume']].describe())
        
        print("\nDate Range:")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End: {df['timestamp'].max()}")
        print(f"Total periods: {len(df)}")

# Add this to your main.py
if __name__ == "__main__":
    # First download the data
    download_and_store_historical_data()
    
    # Then view it
    view_stored_data('BTCUSDT', '1h')
    
    # View other intervals
    view_stored_data('BTCUSDT', '4h')
    view_stored_data('BTCUSDT', '1d')
    
    # View other symbols
    view_stored_data('ETHUSDT', '1h')

# if __name__ == "__main__":
#     # Example of downloading and storing data
#     download_and_store_historical_data()
    
#     # Example of using stored data for simulation
#     simulated_data = simulate_with_real_data()

    # # Set use_simulation=True to use simulated data
    # main(use_simulation=True)

# Example usage
# if __name__ == "__main__":
    # Initialize data manager
    # dm = DataManager()
    
    # # Example of saving data
    # data = {
    #     'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
    #     'open': np.random.random(100) * 1000,
    #     'high': np.random.random(100) * 1000,
    #     'low': np.random.random(100) * 1000,
    #     'close': np.random.random(100) * 1000,
    #     'volume': np.random.random(100) * 100
    # }
    # df = pd.DataFrame(data)
    
    # # Save raw data
    # dm.save_raw_data(df, 'BTCUSDT', '1h')
    
    # # Load data
    # loaded_df = dm.load_data('BTCUSDT', '1h', 
    #                         start_date='2024-01-01',
    #                         end_date='2024-01-05')
    
    # # Get available symbols and intervals
    # symbols = dm.get_available_symbols()
    # intervals = dm.get_available_intervals('BTCUSDT')
    
    # print(f"Available symbols: {symbols}")
    # print(f"Available intervals: {intervals}")