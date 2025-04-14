# data_collector.py
import pandas as pd
from binance.client import Client
from datetime import datetime
import numpy as np
from data_simulator import DataSimulator
from data_manager import DataManager
from typing import Optional

class BinanceCollector:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret) if api_key and api_secret else None
    
    def get_historical_data(self, symbol: str, interval: str, start_str: str, end_str: Optional[str] = None):
        """Fetch historical data from Binance"""
        if not self.client:
            raise ValueError("Binance client not initialized. Please provide API credentials.")
            
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

class DataCollector(BinanceCollector):
    def __init__(self, api_key=None, api_secret=None, use_simulation=False, use_stored_data=False):
        super().__init__(api_key, api_secret)
        self.use_simulation = use_simulation
        self.use_stored_data = use_stored_data
        self.data_manager = DataManager()
        self.simulator = DataSimulator() if use_simulation else None
    
    def get_historical_data(self, symbol: str, interval: str, start_str: str, end_str: Optional[str] = None):
        """Fetch historical data from either Binance, simulation, or stored data"""
        if self.use_stored_data:
            # Try to load from storage first
            data = self.data_manager.load_data(symbol, interval)
            if data is not None:
                return data
            
        if self.use_simulation:
            # Use simulator if no stored data or simulation is specifically requested
            interval_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
            interval_minutes = interval_map.get(interval, 60)
            
            if 'day' in start_str:
                days = int(start_str.split()[0])
            elif 'month' in start_str:
                days = int(start_str.split()[0]) * 30
            else:
                days = 30  # default to 30 days
            
            # Initialize simulator if it's None
            if self.simulator is None:
                self.simulator = DataSimulator()
                
            return self.simulator.generate_historical_data(
                days=days,
                interval_minutes=interval_minutes
            )
        
        # Fetch from Binance and store the data
        data = super().get_historical_data(symbol, interval, start_str, end_str)
        if data is not None:
            self.data_manager.save_raw_data(data, symbol, interval)
        return data