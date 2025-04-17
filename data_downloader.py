"""
Data Downloader Module

This module handles downloading cryptocurrency data from various sources like CoinGecko
and Yahoo Finance. It provides functions to fetch, clean, and store historical price data.
"""

import pandas as pd
from typing import Optional, Dict, Union, List, Tuple


def download_from_coingecko(symbol: str, days: int = 90, interval: str = "hourly") -> Optional[pd.DataFrame]:
    """
    Download cryptocurrency data from CoinGecko API
    
    Args:
        symbol: The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        days: Number of days of data to fetch (max 90 for hourly data, max 365 for daily)
        interval: Time interval ('daily' or 'hourly')
        
    Returns:
        DataFrame with cryptocurrency data or None if download fails
    """
    try:
        # Try importing pycoingecko
        from pycoingecko import CoinGeckoAPI
    except ImportError:
        print("CoinGecko API not installed. Please run: pip install pycoingecko")
        return None
    
    # Common crypto ID mappings
    coin_mappings = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'XRP': 'ripple',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'ADA': 'cardano',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'BNB': 'binancecoin',
        'XLM': 'stellar'
    }
    
    # Extract the base symbol from various formats
    base_symbol = symbol
    if '-USD' in symbol:
        base_symbol = symbol.split('-')[0]
    elif 'USDT' in symbol:
        base_symbol = symbol.replace('USDT', '')
    
    coin_id = coin_mappings.get(base_symbol, base_symbol.lower())
    
    # Initialize CoinGecko API
    cg = CoinGeckoAPI()
    
    # Convert interval to CoinGecko format
    if interval in ['1h', 'hourly']:
        cg_interval = 'hourly'
    else:
        cg_interval = 'daily'
    
    # Limit the days parameter based on interval
    if cg_interval == 'hourly' and days > 90:
        print(f"Warning: Hourly data is limited to 90 days. Limiting request to 90 days.")
        days = 90
    
    print(f"Downloading {coin_id} data from CoinGecko...")
    try:
        # Get market chart data
        data = cg.get_coin_market_chart_by_id(
            id=coin_id,
            vs_currency='usd',
            days=days,
            interval=cg_interval if cg_interval == 'daily' else None  # hourly is default for shorter timeframes
        )
        
        # Check if we got any data
        if not data or 'prices' not in data:
            print(f"No data available for {coin_id}")
            return None
        
        # Create DataFrame
        prices = data['prices']  # [timestamp, price]
        volumes = data['total_volumes']  # [timestamp, volume]
        
        # Extract price data
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
        
        # Extract volume data
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
        
        # Merge the dataframes
        df = pd.merge(df_prices, df_volumes, on='timestamp')
        
        # Set open, high, low values (CoinGecko only provides close prices)
        # For simplicity, use approximation based on close prices
        df['open'] = df['close'].shift(1)
        # For the first row with NaN after shift
        if pd.isna(df['open'].iloc[0]):
            df.loc[0, 'open'] = df['close'].iloc[0]
        
        # Create synthetic high and low
        # Assuming a price variation of around 2% from close price for high and low
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.99
        
        print(f"Downloaded {len(df)} rows of {coin_id} data")
        return df
    except Exception as e:
        print(f"Error downloading {coin_id}: {str(e)}")
        return None


def download_from_yahoo(symbol: str, period: str = "6mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    """
    Download cryptocurrency data from Yahoo Finance
    
    Args:
        symbol: The cryptocurrency ticker symbol (e.g., 'BTC-USD')
        period: The time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Time interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
    Returns:
        DataFrame with cryptocurrency data or None if download fails
    """
    try:
        # Try importing yfinance
        import yfinance as yf
    except ImportError:
        print("Yahoo Finance API not installed. Please run: pip install yfinance")
        return None
    
    # Convert Binance-style symbols to Yahoo Finance format
    yahoo_symbol = symbol
    if 'USDT' in symbol:
        yahoo_symbol = symbol.replace('USDT', '-USD')
    elif not ('USD' in symbol or '-USD' in symbol):
        # Only add -USD if not already there in some form
        yahoo_symbol = f"{symbol}-USD"
    
    # Download data
    print(f"Downloading {yahoo_symbol} data from Yahoo Finance...")
    try:
        data = yf.download(
            tickers=yahoo_symbol,
            period=period,
            interval=interval
        )
        
        # Check if we got any data
        if data is None or data.empty:
            print(f"No data available for {yahoo_symbol}")
            return None
            
        # Reset index to make timestamp a column
        data = data.reset_index()
        
        # Rename columns to match our expected format
        data = data.rename(columns={
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        print(f"Downloaded {len(data)} rows of {yahoo_symbol} data")
        return data
    except Exception as e:
        print(f"Error downloading {yahoo_symbol}: {str(e)}")
        return None


def download_and_store_historical_data(data_manager, symbols: Optional[List[str]] = None) -> None:
    """
    Download and store historical data from CoinGecko
    
    Args:
        data_manager: DataManager instance to handle storage
        symbols: List of symbols to download, defaults to ['BTC'] if None
    """
    if symbols is None:
        symbols = ['BTC']
        
    intervals = ['hourly', 'daily']
    days_list = [90, 365]  # 90 days for hourly, 365 for daily
    
    for symbol in symbols:
        for interval, days in zip(intervals, days_list):
            print(f"Downloading {symbol} {interval} data...")
            df = download_from_coingecko(
                symbol=symbol,
                days=days,
                interval=interval
            )
            
            if df is not None and not df.empty:
                # Convert symbol to Binance format for storage
                storage_symbol = f"{symbol}USDT"
                
                # Convert interval to match storage format
                storage_interval = "1h" if interval == "hourly" else "1d"
                
                # Make sure the dataframe is ready for storage
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure standard numeric columns
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                
                # Save to storage
                try:
                    filepath = data_manager.save_raw_data(df, storage_symbol, storage_interval)
                    print(f"Saved to {filepath}")
                except Exception as e:
                    print(f"Error saving data: {str(e)}")
                    print("Continuing with next dataset...")
            else:
                print(f"No data available to save for {symbol} {interval}") 