# data_manager.py
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime, timedelta
import pytz
import shutil
from typing import Optional, Union, List, Dict

class DataManager:
    def __init__(self, base_dir: str = "crypto_data"):
        """
        Initialize DataManager with directory structure and logging
        
        Parameters:
        -----------
        base_dir : str
            Base directory for all data storage
        """
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir, "raw")
        self.processed_dir = os.path.join(base_dir, "processed")
        self.cache_dir = os.path.join(base_dir, "cache")
        self.metadata_dir = os.path.join(base_dir, "metadata")
        
        # Initialize directory structure
        self._initialize_directories()
        
        # Setup logging
        self._setup_logging()
        
    def _initialize_directories(self):
        """Create necessary directory structure"""
        for directory in [self.base_dir, self.raw_dir, self.processed_dir, 
                         self.cache_dir, self.metadata_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created directory: {directory}")
    
    def _setup_logging(self):
        """Configure logging settings"""
        log_file = os.path.join(self.base_dir, 'data_manager.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def save_raw_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """
        Save raw data with validation and metadata
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing the cryptocurrency data
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Time interval (e.g., '1h', '4h', '1d')
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Validate data
        if not self._validate_dataframe(df):
            raise ValueError("DataFrame validation failed")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{interval}_{timestamp}.csv"
        filepath = os.path.join(self.raw_dir, filename)
        
        # Save data
        df.to_csv(filepath, index=False)
        
        # Create and save metadata
        metadata = self._create_metadata(df, symbol, interval, timestamp)
        self._save_metadata(metadata, filename)
        
        logging.info(f"Saved raw data: {filepath}")
        return filepath
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate DataFrame structure and content
        
        Returns:
        --------
        bool
            True if validation passes, False otherwise
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        if not all(col in df.columns for col in required_columns):
            logging.error("Missing required columns")
            return False
        
        # Check for missing values
        if df[required_columns].isna().any().any():
            logging.error("Found missing values in required columns")
            return False
        
        # Check data types
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
        except Exception as e:
            logging.error(f"Data type conversion failed: {e}")
            return False
        
        return True
    
    def _create_metadata(self, df: pd.DataFrame, symbol: str, interval: str, 
                        timestamp: str) -> Dict:
        """Create metadata dictionary for the dataset"""
        # Convert missing values dict to ensure all keys are strings
        missing_values_dict = {}
        for key, value in df.isna().sum().items():
            # Convert any non-string keys to strings
            missing_values_dict[str(key)] = int(value)
        
        # Safely get min and max timestamps
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Convert timestamps to strings in a safe manner
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_date = str(start_date)
        
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")
        else:
            end_date = str(end_date)
        
        return {
            "symbol": symbol,
            "interval": interval,
            "timestamp": timestamp,
            "rows": len(df),
            "start_date": start_date,
            "end_date": end_date,
            "columns": [str(col) for col in df.columns],  # Ensure all column names are strings
            "missing_values": missing_values_dict,
            "file_version": "1.0"
        }
    
    def _save_metadata(self, metadata: Dict, data_filename: str):
        """Save metadata to JSON file"""
        metadata_filename = data_filename.replace('.csv', '_metadata.json')
        metadata_path = os.path.join(self.metadata_dir, metadata_filename)
        
        # Ensure all values are JSON serializable
        def json_serializable(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Try standard conversion
                return str(obj)
            except Exception:
                return "Non-serializable object"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=json_serializable)
            logging.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            # If serialization still fails, log error and create simplified metadata
            logging.error(f"Error saving metadata: {str(e)}")
            simplified_metadata = {
                "symbol": metadata.get("symbol", "unknown"),
                "interval": metadata.get("interval", "unknown"),
                "timestamp": metadata.get("timestamp", "unknown"),
                "rows": metadata.get("rows", 0),
                "file_version": "1.0"
            }
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(simplified_metadata, f, indent=4)
                logging.info(f"Simplified metadata saved to {metadata_path}")
            except Exception as e2:
                logging.error(f"Failed to save even simplified metadata: {str(e2)}")
    
    def load_data(self, symbol: str, interval: str, 
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None) -> Optional[pd.DataFrame]:
        """
        Load data for given symbol and interval within date range
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol
        interval : str
            Time interval
        start_date : str or datetime, optional
            Start date in 'YYYY-MM-DD' format or as datetime object
        end_date : str or datetime, optional
            End date in 'YYYY-MM-DD' format or as datetime object
            
        Returns:
        --------
        pandas.DataFrame or None
            Loaded data or None if no data found
        """
        # Convert dates if provided
        if start_date:
            start_date = pd.to_datetime(start_date)
        if end_date:
            end_date = pd.to_datetime(end_date)
        
        # Get list of relevant files
        files = self._get_data_files(symbol, interval)
        
        if not files:
            logging.warning(f"No data files found for {symbol} {interval}")
            return None
        
        # Load and combine data
        dfs = []
        for file in files:
            try:
                df = pd.read_csv(os.path.join(self.raw_dir, file))
                
                # Convert timestamp to datetime and handle timezone issues
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Make all timestamps timezone-naive by converting timezone-aware ones
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                
                # Apply date filters if provided
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                    
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error loading file {file}: {str(e)}")
                continue
        
        # Combine all dataframes
        if dfs:
            try:
                combined_df = pd.concat(dfs)
                
                # Ensure all timestamps are timezone-naive before sorting
                if combined_df['timestamp'].dt.tz is not None:
                    combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize(None)
                    
                # Drop duplicates and sort
                combined_df = combined_df.drop_duplicates().sort_values('timestamp')
                return combined_df
            except Exception as e:
                logging.error(f"Error combining dataframes: {str(e)}")
                # If combining fails, return the most recent dataset
                if dfs:
                    return dfs[-1]
        
        return None
    
    def _get_data_files(self, symbol: str, interval: str) -> List[str]:
        """Get list of data files for given symbol and interval"""
        return [f for f in os.listdir(self.raw_dir) 
                if f.startswith(f"{symbol}_{interval}") and f.endswith('.csv')]
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs"""
        files = os.listdir(self.raw_dir)
        symbols = set()
        for file in files:
            if file.endswith('.csv'):
                symbol = file.split('_')[0]
                symbols.add(symbol)
        return sorted(list(symbols))
    
    def get_available_intervals(self, symbol: str) -> List[str]:
        """Get list of available intervals for a symbol"""
        files = self._get_data_files(symbol, '')
        intervals = set()
        for file in files:
            interval = file.split('_')[1]
            intervals.add(interval)
        return sorted(list(intervals))
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for directory in [self.raw_dir, self.processed_dir, self.cache_dir]:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                file_timestamp = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_timestamp < cutoff_date:
                    os.remove(filepath)
                    logging.info(f"Removed old file: {filepath}")