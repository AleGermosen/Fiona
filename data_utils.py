"""
Data Utilities Module

This module provides utilities for data cleaning, processing, and management
of cryptocurrency data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Any


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a cryptocurrency dataframe by removing invalid data points.
    
    Args:
        df: Input DataFrame with cryptocurrency data
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        print("No data to clean")
        return df
    
    print("Cleaning data...")
    try:
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Remove rows with NaT timestamps
        df_clean = df_clean[~pd.isna(df_clean['timestamp'])]
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Sort by timestamp
        df_clean = df_clean.sort_values('timestamp')
        
        # Convert string prices to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df_clean.columns and df_clean[col].dtype == 'object':
                # First remove rows with problematic values
                if hasattr(df_clean[col], 'str'):
                    invalid_mask = df_clean[col].str.contains('USD', na=False, regex=True)
                    if invalid_mask.any():
                        df_clean = df_clean[~invalid_mask]
                
                # Convert remaining strings to float
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                # Fill NaN values with forward fill then backward fill
                df_clean[col] = df_clean[col].ffill().bfill()
        
        # Final check - remove any remaining NaN values in timestamp and close
        df_clean = df_clean.dropna(subset=['timestamp', 'close'])
        
        return df_clean
    
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        return df  # Return original if cleaning fails


def evaluate_data_quality(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate the quality of a cryptocurrency dataset.
    
    Args:
        df: DataFrame with cryptocurrency data
        
    Returns:
        Tuple containing (score, details_dict) where:
            score: Data quality score from 0-100 (higher is better)
            details_dict: Dictionary with detailed quality metrics
    """
    if df is None or df.empty:
        return 0, {"error": "Empty dataset"}
    
    # Start with a perfect score
    score = 100
    details = {}
    
    # Check for NaT timestamps
    nat_pct = df['timestamp'].isna().mean() * 100
    score -= nat_pct
    details["missing_timestamps_pct"] = nat_pct
    
    # Check for numeric columns
    numeric_issues = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            # Check if column is already numeric
            if df[col].dtype != 'object':
                numeric_issues[col] = 0
                continue
                
            # Check for problematic string values
            if hasattr(df[col], 'str'):
                invalid_pct = df[col].str.contains('USD', na=False, regex=True).mean() * 100
                score -= invalid_pct
                numeric_issues[col] = invalid_pct
            
            # Try converting to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
            except:
                score -= 10
                numeric_issues[col] = "not_convertible"
    
    details["numeric_issues"] = numeric_issues
    
    # Check for duplicates
    duplicate_pct = df.duplicated(subset=['timestamp']).mean() * 100
    score -= duplicate_pct
    details["duplicate_timestamps_pct"] = duplicate_pct
    
    # Check for chronological ordering
    is_sorted = (df['timestamp'].diff().dropna() >= pd.Timedelta(0)).all()
    if not is_sorted:
        score -= 5
    details["chronologically_ordered"] = is_sorted
    
    # Check for large gaps
    time_diffs = df['timestamp'].diff().dropna()
    if len(time_diffs) > 0:
        # Get expected interval from median time difference
        # Convert timedelta to numeric seconds
        time_diffs_seconds = time_diffs.dt.total_seconds()  # Fixed line
        expected_interval_sec = time_diffs_seconds.median()
        
        # Count gaps more than 2x the expected interval
        large_gaps = (time_diffs_seconds > 2 * expected_interval_sec).sum()
        large_gap_pct = (large_gaps / len(time_diffs)) * 100
        score -= min(10, large_gap_pct)  # Cap penalty at 10 points
        details["large_gaps_pct"] = large_gap_pct
    
    # Check data volume (more data points is better)
    row_bonus = min(10, len(df) / 100)  # Up to 10 points bonus for row count
    score += row_bonus
    details["data_points"] = len(df)
    details["data_volume_bonus"] = row_bonus
    
    # Cap the score between 0 and 100
    score = max(0, min(100, score))
    details["final_score"] = score
    
    return score, details


def prepare_data_for_modeling(df: pd.DataFrame, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Prepare data for modeling by filtering by date and cleaning.
    
    Args:
        df: DataFrame with cryptocurrency data
        start_date: Optional start date to filter data (format: 'YYYY-MM-DD')
        
    Returns:
        Prepared DataFrame ready for modeling
    """
    # Clean the data first
    df = clean_dataframe(df)
    
    # Filter by date if specified
    if start_date and not df.empty:
        start_timestamp = pd.to_datetime(start_date)
        df = df[df['timestamp'] >= start_timestamp].reset_index(drop=True)
        print(f"Filtered data from {start_timestamp}, {len(df)} rows remaining")
    
    return df


def remove_outliers(df: pd.DataFrame, columns: List[str] = ['open', 'high', 'low', 'close'], method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from dataframe columns using statistical methods.
    
    Args:
        df: DataFrame with cryptocurrency data
        columns: List of column names to check for outliers
        method: Method to use ('zscore' or 'iqr')
        threshold: Threshold for outlier detection (z-score or IQR multiplier)
        
    Returns:
        DataFrame with outliers removed
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    original_len = len(df_clean)
    
    # Process each specified column
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        # Ensure column is numeric
        if df_clean[col].dtype == 'object':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Skip if column has all NaN values
        if df_clean[col].isna().all():
            continue
            
        if method == 'zscore':
            # Z-score method
            from scipy import stats
            z_scores = stats.zscore(df_clean[col], nan_policy='omit')
            abs_z_scores = np.abs(z_scores)
            outlier_mask = abs_z_scores > threshold
            df_clean = df_clean[~outlier_mask]
            
        elif method == 'iqr':
            # IQR method
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    # Check how many rows were removed
    removed = original_len - len(df_clean)
    if removed > 0:
        print(f"Removed {removed} outliers ({removed/original_len:.1%} of data)")
    
    return df_clean


def find_cleanest_dataset(data_manager, symbol: str = 'BTCUSDT', 
                        preferred_interval: str = '1h') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Find and clean the best dataset available for a given symbol,
    prioritizing recent data and removing outliers.
    
    Args:
        data_manager: DataManager instance to handle data loading
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        preferred_interval: The preferred interval to use if available
        
    Returns:
        Tuple containing (DataFrame, interval) with the cleaned dataset and its interval
    """
    # Get available intervals
    intervals = data_manager.get_available_intervals(symbol)
    if not intervals:
        print(f"No data available for {symbol}")
        return None, None
    
    # If preferred interval exists, prioritize it
    if preferred_interval in intervals:
        intervals.remove(preferred_interval)
        intervals.insert(0, preferred_interval)
    
    # Load and evaluate all datasets
    datasets = {}
    for interval in intervals:
        df = data_manager.load_data(symbol, interval)
        if df is None or df.empty:
            continue
        
        # Get the date range
        start_date = df['timestamp'].min()
        end_date = df['timestamp'].max()
        
        # Evaluate data quality
        score, details = evaluate_data_quality(df)
        
        # Clean the data and remove outliers
        df_clean = clean_dataframe(df)
        df_clean = remove_outliers(df_clean)
        
        # Store the cleaned dataset with metadata
        datasets[interval] = {
            'df': df_clean,
            'score': score,
            'start_date': start_date,
            'end_date': end_date,
            'row_count': len(df_clean)
        }
        
        print(f"Interval {interval}: score {score:.1f}, {len(df_clean)} clean rows, date range: {start_date.strftime('%Y-%m-%d %H:%M:%S')} to {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not datasets:
        print("No valid datasets found")
        return None, None
        
    # Find most recent data across all datasets
    most_recent_date = max([info['end_date'] for info in datasets.values()])
    print(f"Most recent date across all datasets: {most_recent_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define threshold for recency: data must be from the last 30 days to be considered "recent"
    recency_threshold = most_recent_date - pd.Timedelta(days=30)
    print(f"Using recency threshold: {recency_threshold.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find datasets with recent data
    recent_datasets = {
        interval: info for interval, info in datasets.items() 
        if info['end_date'] >= recency_threshold
    }
    
    if recent_datasets:
        print(f"Found {len(recent_datasets)} datasets with recent data")
        
        # If preferred interval is in recent datasets, use it
        if preferred_interval in recent_datasets:
            print(f"Using preferred interval {preferred_interval} with recent data")
            return recent_datasets[preferred_interval]['df'], preferred_interval
        
        # Otherwise find the dataset with the most recent end date
        most_recent_interval = max(recent_datasets.items(), key=lambda x: x[1]['end_date'])[0]
        print(f"Using most recent dataset: {most_recent_interval}, end date: {recent_datasets[most_recent_interval]['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
        return recent_datasets[most_recent_interval]['df'], most_recent_interval
    
    # If no dataset meets the recency requirement, 
    # strongly favor recency over data quality
    print("No datasets meet the recency threshold, falling back to most recent dataset")
    most_recent_interval = max(datasets.items(), key=lambda x: x[1]['end_date'])[0]
    print(f"Using most recent dataset: {most_recent_interval}, end date: {datasets[most_recent_interval]['end_date'].strftime('%Y-%m-%d %H:%M:%S')}")
    return datasets[most_recent_interval]['df'], most_recent_interval


def list_available_data(data_manager) -> None:
    """
    List all available cryptocurrency data in storage.
    
    Args:
        data_manager: DataManager instance to handle data listing
    """
    print("\n=== Available Cryptocurrency Data ===\n")
    
    symbols = data_manager.get_available_symbols()
    if not symbols:
        print("No data found in storage.")
        return
    
    print(f"Found {len(symbols)} symbols: {', '.join(symbols)}")
    
    for symbol in symbols:
        intervals = data_manager.get_available_intervals(symbol)
        print(f"\n{symbol} available intervals: {', '.join(intervals)}")
        
        for interval in intervals:
            # Try to get file count
            files = data_manager._get_data_files(symbol, interval)
            print(f"  - {interval}: {len(files)} file(s)")
            
            # Load a sample and show structure
            try:
                df = data_manager.load_data(symbol, interval)
                if df is not None and not df.empty:
                    print(f"    * {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
                    print(f"    * Data types: {df.dtypes.to_dict()}")
                    print(f"    * First few values: {df['close'].head(3).values}")
            except Exception as e:
                print(f"    * Error inspecting data: {str(e)}")


def view_stored_data(data_manager, symbol: str = 'BTCUSDT', interval: str = '1h', 
                   rows: int = 5, include_stats: bool = True, 
                   use_stored_only: bool = True) -> Optional[pd.DataFrame]:
    """
    View stored cryptocurrency data with optional statistics.
    
    Args:
        data_manager: DataManager instance to handle data loading
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        rows: Number of rows to display from start and end
        include_stats: Whether to show basic statistics
        use_stored_only: Whether to only use stored data without downloading
        
    Returns:
        The loaded DataFrame or None if not available
    """
    from data_downloader import download_from_coingecko
    
    df = data_manager.load_data(symbol, interval)
    
    if df is None and not use_stored_only:
        print(f"No stored data found for {symbol} at {interval} interval. Attempting to download...")
        
        # Extract the base symbol
        base_symbol = symbol.replace("USDT", "")
        
        # Determine the appropriate interval for CoinGecko
        cg_interval = "hourly" if interval in ["1h", "4h"] else "daily"
        days = 90 if cg_interval == "hourly" else 365
            
        # Try to download data
        df = download_from_coingecko(base_symbol, days=days, interval=cg_interval)
        
        if df is not None and not df.empty:
            # Save the downloaded data
            filepath = data_manager.save_raw_data(df, symbol, interval)
            print(f"Downloaded and saved data to {filepath}")
    
    if df is None:
        print(f"No data found for {symbol} at {interval} interval")
        return None
    
    # Show raw data first
    print(f"\n=== {symbol} {interval} Raw Data ===")
    print(f"Total rows: {len(df)}")
    print(f"Column types: {df.dtypes}")
    
    # Check for problematic data
    issues = []
    if df['timestamp'].isna().any():
        issues.append(f"- {df['timestamp'].isna().sum()} rows with NaT timestamps")
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            if df[col].dtype == 'object':
                invalid_mask = df[col].str.contains('USD', na=False, regex=True)
                if invalid_mask.any():
                    issues.append(f"- {invalid_mask.sum()} rows with 'USD' in {col} column")
    
    if issues:
        print("\nData issues detected:")
        for issue in issues:
            print(issue)
        
        # Attempt to clean the data
        print("\nCleaning data...")
        df_clean = clean_dataframe(df)
        
        if len(df_clean) < len(df):
            print(f"After cleaning: {len(df_clean)} valid rows remaining")
            df = df_clean
        
    print(f"\n=== {symbol} {interval} Cleaned Data ===")
    
    # Show first few rows
    print(f"\nFirst {rows} rows:")
    print(df.head(rows))
    
    # Show last few rows
    print(f"\nLast {rows} rows:")
    print(df.tail(rows))
    
    if include_stats:
        # For statistics, ensure numeric columns are properly converted
        stats_df = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in stats_df.columns:
                stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        print("\nBasic Statistics:")
        print(stats_df[['open', 'high', 'low', 'close', 'volume']].describe())
        
        print("\nDate Range:")
        print(f"Start: {df['timestamp'].min()}")
        print(f"End: {df['timestamp'].max()}")
        print(f"Total valid periods: {len(df)}")
    
    return df 


def join_timeframe_datasets(data_manager, symbol: str = 'BTCUSDT', 
                          intervals: List[str] = ['1h', '4h', '1d']) -> Optional[pd.DataFrame]:
    """
    Join data from multiple timeframes into a single dataset for a given symbol.
    
    This function loads data for each specified interval, combines them into a single 
    DataFrame, removes duplicates, and ensures all data is properly sorted by timestamp.
    
    Args:
        data_manager: DataManager instance to handle data loading
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        intervals: List of time intervals to join (e.g., ['1h', '4h', '1d'])
        
    Returns:
        Combined DataFrame with data from all specified intervals, or None if no data is available
    """
    all_dfs = []
    
    for interval in intervals:
        df = data_manager.load_data(symbol, interval)
        if df is not None and not df.empty:
            print(f"Loaded {len(df)} rows of {symbol} data at {interval} interval")
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Add interval info to distinguish source in case it's needed later
            df['source_interval'] = interval
            all_dfs.append(df)
        else:
            print(f"No data available for {symbol} at {interval} interval")
    
    if not all_dfs:
        print(f"No data available for {symbol} across any of the specified intervals")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Ensure all timestamps are timezone-naive before processing
    if combined_df['timestamp'].dt.tz is not None:
        combined_df['timestamp'] = combined_df['timestamp'].dt.tz_localize(None)
    
    # Drop duplicates based on timestamp
    # If multiple intervals have the same timestamp, keep the one with the smaller interval
    # This prioritizes more granular data (e.g., 1h over 4h over 1d)
    combined_df = combined_df.sort_values(['timestamp', 'source_interval'])
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
    
    # Sort by timestamp for final dataset
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Combined dataset has {len(combined_df)} rows after removing duplicates")
    return combined_df 