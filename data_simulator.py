# data_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from feature_engineer import FeatureEngineer
from data_preprocessor import DataPreprocessor
from predictor import PricePredictor
from model_builder import ModelBuilder
from typing import Optional

class DataSimulator:
    def __init__(self, starting_price=30000, volatility=0.02, trend=0):
        """
        Initialize the simulator
        starting_price: Initial price of the asset
        volatility: Price volatility (0.02 = 2%)
        trend: Price trend (-0.01 = downtrend, 0 = neutral, 0.01 = uptrend)
        """
        self.current_price = starting_price
        self.volatility = volatility
        self.trend = trend

    def generate_historical_data(self, days=30, interval_minutes=60):
        """Generate simulated historical data"""
        periods = int((days * 24 * 60) / interval_minutes)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Generate timestamps
        timestamps = pd.date_range(start=start_time, end=end_time, periods=periods)
        
        # Generate prices using geometric Brownian motion
        returns = np.random.normal(loc=self.trend, scale=self.volatility, size=periods)
        price_factors = np.exp(returns).cumprod()
        prices = self.current_price * price_factors
        
        # Generate volume with some randomness
        base_volume = self.current_price * 100  # Base daily volume
        volumes = np.random.normal(loc=base_volume, scale=base_volume*0.1, size=periods)
        volumes = np.abs(volumes)  # Ensure positive volumes
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.01, periods)),
            'low': prices * (1 - np.random.uniform(0, 0.01, periods)),
            'close': prices,
            'volume': volumes,
            'quote_volume': volumes * prices,
            'trades': np.random.randint(1000, 5000, periods),
            'taker_base': volumes * 0.4,  # Assume 40% taker volume
            'taker_quote': volumes * prices * 0.4,
        })
        
        # Ensure OHLC relationship is maintained
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)
        
        return df

# modified data_collector.py
class DataCollector:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, use_simulation: bool = False):
        self.use_simulation = use_simulation
        if not use_simulation and (api_key and api_secret):
            self.client = Client(api_key, api_secret)
        self.simulator = DataSimulator() if use_simulation else None
    
    def get_historical_data(self, symbol: str, interval: str, start_str: str, end_str: Optional[str] = None):
        """Fetch historical data from either Binance or simulation"""
        if self.use_simulation:
            # Convert interval to minutes for simulator
            interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
            interval_minutes = interval_map.get(interval, 60)
            
            # Convert start_str to number of days
            if 'day' in start_str:
                days = int(start_str.split()[0])
            elif 'month' in start_str:
                days = int(start_str.split()[0]) * 30
            else:
                days = 30  # default to 30 days
            
            # Initialize simulator if it's None
            if self.simulator is None:
                self.simulator = DataSimulator()
                
            df = self.simulator.generate_historical_data(
                days=days,
                interval_minutes=interval_minutes
            )
            return df
        else:
            # Original Binance data collection code
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
            
            return df

# Modified main.py with simulation support
def main(use_simulation=True):
    # Initialize components
    api_key = "YOUR_API_KEY" if not use_simulation else None
    api_secret = "YOUR_API_SECRET" if not use_simulation else None
    
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

if __name__ == "__main__":
    # Set use_simulation=True to use simulated data
    main(use_simulation=True)