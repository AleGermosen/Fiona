# data_storage.py
import pandas as pd
import os
import json
from datetime import datetime
import pickle

class DataManager:
    def __init__(self, storage_dir="crypto_data"):
        """
        Initialize DataManager with a storage directory
        """
        self.storage_dir = storage_dir
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            os.makedirs(os.path.join(self.storage_dir, "raw"))
            os.makedirs(os.path.join(self.storage_dir, "processed"))
    
    def save_data(self, df, symbol, interval, data_type="raw"):
        """
        Save data to CSV with metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{interval}_{timestamp}.csv"
        filepath = os.path.join(self.storage_dir, data_type, filename)
        
        # Save the DataFrame
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            "symbol": symbol,
            "interval": interval,
            "timestamp": timestamp,
            "rows": len(df),
            "start_date": df['timestamp'].min(),
            "end_date": df['timestamp'].max()
        }
        
        metadata_path = filepath.replace(".csv", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=str)
        
        return filepath
    
    def load_latest_data(self, symbol, interval, data_type="raw"):
        """
        Load the most recent data file for given symbol and interval
        """
        directory = os.path.join(self.storage_dir, data_type)
        files = [f for f in os.listdir(directory) if f.endswith('.csv') 
                and f.startswith(f"{symbol}_{interval}")]
        
        if not files:
            return None
        
        latest_file = max(files)
        filepath = os.path.join(directory, latest_file)
        return pd.read_csv(filepath)
    
    def load_data_by_date(self, symbol, interval, start_date, end_date, data_type="raw"):
        """
        Load data within specified date range
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        directory = os.path.join(self.storage_dir, data_type)
        all_data = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.csv') and filename.startswith(f"{symbol}_{interval}"):
                df = pd.read_csv(os.path.join(directory, filename))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
                all_data.append(df[mask])
        
        if not all_data:
            return None
        
        return pd.concat(all_data).drop_duplicates().sort_values('timestamp')
