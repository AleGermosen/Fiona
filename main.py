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
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Load environment variables from .env file
load_dotenv()

# Function to download data from Yahoo Finance
def download_from_yahoo(symbol, period="6mo", interval="1h"):
    """
    Download cryptocurrency data from Yahoo Finance
    
    Args:
        symbol (str): The cryptocurrency ticker symbol (e.g., 'BTC-USD')
        period (str): The time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Time interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
    Returns:
        pandas.DataFrame: DataFrame with cryptocurrency data
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

def main(use_simulation=True):
    # Initialize components
    api_key = os.getenv("API_KEY") if not use_simulation else None
    api_secret = os.getenv("API_SECRET") if not use_simulation else None
    
    # Create instances
    collector = DataCollector(api_key, api_secret, use_simulation=use_simulation)
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    # Collect and prepare data
    if use_simulation:
        # Use simulated data
        simulator = DataSimulator()
        df = simulator.generate_historical_data()
    else:
        # Try to get data from Yahoo Finance
        df = download_from_yahoo(symbol="BTC-USD", period="6mo", interval="1h")
        
        # Fallback to simulation if Yahoo Finance fails
        if df is None:
            print("Falling back to simulated data...")
            simulator = DataSimulator()
            df = simulator.generate_historical_data()
    
    df = engineer.add_technical_indicators(df)
    X, y = preprocessor.prepare_data(df)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = ModelBuilder.build_lstm_model(
        sequence_length=60,
        n_features=9
    )
    
    # Train the model
    print("Training prediction model...")
    
    # Check for NaN or infinite values in training data
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Warning: Training data contains NaN or infinite values. Cleaning data...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        print("Warning: Target data contains NaN or infinite values. Cleaning data...")
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
    # Use a lower learning rate and add gradient clipping
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    
    # Compile model with lower learning rate and gradient clipping
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='mse'
    )
    
    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
        
    history = model.fit(
        X_train, y_train,
        epochs=20,  # Reduced for faster execution
        batch_size=32,
        validation_split=0.1,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    predictor = PricePredictor(model, preprocessor)
    last_sequence = X_test[-1:]
    next_price = predictor.predict_next_price(last_sequence)
    print(f"Predicted next price: ${next_price:.2f}")

# Comment out the original download function or modify it to use Yahoo Finance
def download_and_store_historical_data():
    """Script to download and store historical data from Yahoo Finance"""
    data_manager = DataManager()
    
    # Define what data you want to download
    symbols = ['BTC-USD', 'ETH-USD']
    intervals = ['1h', '1d']
    periods = ['6mo', '1y']
    
    for symbol in symbols:
        for interval, period in zip(intervals, periods):
            print(f"Downloading {symbol} {interval} data...")
            df = download_from_yahoo(
                symbol=symbol,
                period=period,
                interval=interval
            )
            
            if df is not None and not df.empty:
                # Convert Yahoo symbol to Binance format for storage
                if '-USD' in symbol:
                    storage_symbol = symbol.replace('-USD', 'USDT')
                else:
                    storage_symbol = symbol
                
                # Make sure the dataframe is ready for storage
                # This ensures timestamps are correctly formatted and all columns have proper data types
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Ensure standard numeric columns
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                
                # Save to storage
                try:
                    filepath = data_manager.save_raw_data(df, storage_symbol, interval)
                    print(f"Saved to {filepath}")
                except Exception as e:
                    print(f"Error saving data: {str(e)}")
                    print("Continuing with next dataset...")
            else:
                print(f"No data available to save for {symbol} {interval}")

# Example of using stored or simulated data
def simulate_with_real_data():
    """Example of using stored data for simulation"""
    data_manager = DataManager()
    
    # Try to load historical data
    df = data_manager.load_data("BTCUSDT", "1h")
    
    # If no stored data, try getting from Yahoo Finance
    if df is None:
        df = download_from_yahoo("BTC-USD", period="6mo", interval="1h")
        
    # If still no data, use simulation
    if df is None:
        simulator = DataSimulator()
        simulated_df = simulator.generate_historical_data()
        return simulated_df
    else:
        # Create simulator with parameters based on real data
        simulator = DataSimulator(
            starting_price=df['close'].iloc[-1],
            volatility=df['close'].pct_change().std(),
            trend=int(df['close'].pct_change().mean())
        )
        
        # Generate simulated data with characteristics of real data
        simulated_df = simulator.generate_historical_data()
        return simulated_df

def view_stored_data(symbol='BTCUSDT', interval='1h', rows=5, include_stats=True, use_stored_only=True):
    """
    View stored cryptocurrency data with optional statistics.
    
    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval (e.g., '1h', '4h', '1d')
        rows (int): Number of rows to display from start and end
        include_stats (bool): Whether to show basic statistics
        use_stored_only (bool): Whether to only use stored data without downloading
    """
    data_manager = DataManager()
    df = data_manager.load_data(symbol, interval)
    
    if df is None and not use_stored_only:
        print(f"No stored data found for {symbol} at {interval} interval. Attempting to download...")
        
        # Convert symbol format if needed
        yahoo_symbol = symbol
        if 'USDT' in symbol:
            yahoo_symbol = symbol.replace('USDT', '-USD')
            
        # Try to download data
        df = download_from_yahoo(yahoo_symbol, period="6mo", interval=interval)
        
        if df is not None and not df.empty:
            # Save the downloaded data
            filepath = data_manager.save_raw_data(df, symbol, interval)
            print(f"Downloaded and saved data to {filepath}")
    
    if df is None:
        print(f"No data found for {symbol} at {interval} interval")
        return
    
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
        try:
            # Remove rows with NaT timestamps
            df_clean = df[~pd.isna(df['timestamp'])]
            
            # Remove duplicates
            df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # Sort by timestamp
            df_clean = df_clean.sort_values('timestamp')
            
            # Convert string prices to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_clean.columns and df_clean[col].dtype == 'object':
                    # First remove rows with problematic values
                    invalid_mask = df_clean[col].str.contains('USD', na=False, regex=True)
                    if invalid_mask.any():
                        df_clean = df_clean[~invalid_mask]
                    
                    # Convert remaining strings to float
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Fill NaN values with forward fill then backward fill
                    df_clean[col] = df_clean[col].ffill().bfill()
            
            # Final check - remove any remaining NaN values in timestamp and close
            df_clean = df_clean.dropna(subset=['timestamp', 'close'])
            
            print(f"After cleaning: {len(df_clean)} valid rows remaining")
            
            # Use the cleaned dataframe
            df = df_clean
        except Exception as e:
            print(f"Error during cleaning: {str(e)}")
            print("Showing raw data instead.")
    
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

def plot_crypto_data(symbol='BTCUSDT', interval='1h', start_date=None, end_date=None, use_stored_only=True):
    """
    Plot cryptocurrency price data with volume.
    
    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval (e.g., '1h', '4h', '1d')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        use_stored_only (bool): Whether to only use stored data without downloading
    """
    # Try to load stored data first
    data_manager = DataManager()
    df = data_manager.load_data(symbol, interval)
    
    # If no stored data and allowed to download, try to download it
    if (df is None or df.empty) and not use_stored_only:
        print(f"No stored data found for {symbol} at {interval} interval. Attempting to download...")
        
        # Convert symbol format if needed
        yahoo_symbol = symbol
        if 'USDT' in symbol:
            yahoo_symbol = symbol.replace('USDT', '-USD')
        elif not ('USD' in symbol or '-USD' in symbol):
            # Only add -USD if not already there in some form
            yahoo_symbol = f"{symbol}-USD"
            
        # Try to download data
        df = download_from_yahoo(yahoo_symbol, period="6mo", interval=interval)
        
        if df is None or df.empty:
            print(f"Could not get data for {symbol}. Generating simulated data instead.")
            simulator = DataSimulator()
            df = simulator.generate_historical_data()
    elif df is None and use_stored_only:
        print(f"No stored data found for {symbol} and use_stored_only is set to True.")
        print("Set use_stored_only=False to attempt downloading or using simulation.")
        return
    
    # Verify we have data to plot
    if df is None or df.empty:
        print("Error: No data available to plot.")
        return
    
    print("Cleaning data for plotting...")
    
    # Clean the data
    try:
        # Remove rows with NaT timestamps
        df = df[~pd.isna(df['timestamp'])]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Convert string prices to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if df[col].dtype == 'object':
                # First remove rows with problematic values
                invalid_mask = df[col].str.contains('USD', na=False, regex=True)
                if invalid_mask.any():
                    print(f"Removing {invalid_mask.sum()} rows with invalid {col} values (containing 'USD')")
                    df = df[~invalid_mask]
                
                # Convert remaining strings to float
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill NaN values with forward fill then backward fill
                df[col] = df[col].ffill().bfill()
    
        # Final check - remove any remaining NaN values
        df = df.dropna(subset=['timestamp', 'close'])
        
        print(f"After cleaning: {len(df)} rows remaining")
        
        if len(df) < 2:
            print("Not enough valid data points to plot after cleaning.")
            return
            
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        print("Attempting to use simulation data instead.")
        simulator = DataSimulator()
        df = simulator.generate_historical_data()
    
    # Filter by date range if provided
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    
    # Debug information
    print(f"Data shape: {df.shape}")
    print(f"First timestamp: {df['timestamp'].iloc[0]}")
    print(f"Last timestamp: {df['timestamp'].iloc[-1]}")
    print(f"First close price: {df['close'].iloc[0]}")
    print(f"Last close price: {df['close'].iloc[-1]}")
    
    # Add technical indicators for prediction
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    
    # Prepare data for prediction
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    if len(X) > 0:
        # Build and train model
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the model
        model = ModelBuilder.build_lstm_model(
            sequence_length=60,
            n_features=X.shape[2]
        )
        
        # Train the model
        print("Training prediction model...")
        
        # Check for NaN or infinite values in training data
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("Warning: Training data contains NaN or infinite values. Cleaning data...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if np.isnan(y_train).any() or np.isinf(y_train).any():
            print("Warning: Target data contains NaN or infinite values. Cleaning data...")
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Use a lower learning rate and add gradient clipping
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
        
        # Compile model with lower learning rate and gradient clipping
        model.compile(
            optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
            loss='mse'
        )
        
        # Add learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
            
        history = model.fit(
            X_train, y_train,
            epochs=20,  # Reduced for faster execution
            batch_size=32,
            validation_split=0.1,
            callbacks=[reduce_lr],
            verbose=1
        )
        
        # Make predictions
        predictor = PricePredictor(model, preprocessor)
        
        # Predict prices for test data
        predicted_prices = []
        for i in range(len(X_test)):
            predicted_price = predictor.predict_next_price(X_test[i:i+1])
            predicted_prices.append(predicted_price)
        
        # Create prediction dataframe
        pred_indices = range(train_size + 60, train_size + 60 + len(predicted_prices))
        pred_dates = df['timestamp'].iloc[pred_indices]
        
        # Make future predictions (next 30 points)
        future_predictions = []
        last_sequence = X[-1:]
        
        print("Generating future predictions...")
        for _ in range(30):
            # Predict next price
            next_pred = predictor.predict_next_price(last_sequence)
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction (simple approach)
            # Create a copy of the last sequence
            new_seq = last_sequence[0].copy()
            # Shift the sequence by one step
            new_seq[:-1] = new_seq[1:]
            # Add the new prediction as the last value in the sequence (for the 'close' price feature)
            new_seq[-1, 0] = y[-1]  # Use the scaled version of the prediction
            
            # Update last_sequence
            last_sequence = np.array([new_seq])
        
        # Generate future dates
        last_date = df['timestamp'].iloc[-1]
        
        # Determine the time interval in hours
        interval_hours = 1
        if interval == '4h':
            interval_hours = 4
        elif interval == '1d':
            interval_hours = 24
            
        future_dates = pd.date_range(
            start=last_date, 
            periods=31, 
            freq=f'{interval_hours}H'
        )[1:]  # Skip the first one which is the last actual data point
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot price data
    ax1.plot(df['timestamp'], df['close'], label='Historical Price', color='blue')
    ax1.set_title(f'{symbol} Price Chart ({interval})')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True)
    
    # Add predictions to the plot if available
    if len(X) > 0:
        # Plot test predictions in red
        ax1.plot(pred_dates, predicted_prices, label='Model Prediction', color='red', linestyle='--')
        
        # Plot future predictions in red
        ax1.plot(future_dates, future_predictions, label='Future Prediction', color='red')
        
        # Add vertical line to mark where predictions start
        if len(pred_dates) > 0:
            ax1.axvline(x=pred_dates.iloc[0], color='gray', linestyle='--', alpha=0.7)
        
        # Add vertical line to mark where future predictions start
        ax1.axvline(x=future_dates[0], color='gray', linestyle='--', alpha=0.7)
    
    ax1.legend()
    
    # Plot volume
    ax2.bar(df['timestamp'], df['volume'], color='green', alpha=0.5)
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def find_cleanest_dataset(symbol='BTCUSDT', preferred_interval='4h'):
    """
    Find the cleanest dataset available for a given symbol
    
    This function examines all available intervals for a symbol and
    returns the dataset with the fewest data issues.
    
    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
        preferred_interval (str): The preferred interval if multiple clean datasets exist
        
    Returns:
        tuple: (DataFrame, interval) containing the cleaned data and the interval used
    """
    data_manager = DataManager()
    
    # Get available intervals
    intervals = data_manager.get_available_intervals(symbol)
    if not intervals:
        print(f"No data available for {symbol}")
        return None, None
    
    # If preferred interval exists, start with that
    if preferred_interval in intervals:
        intervals.remove(preferred_interval)
        intervals.insert(0, preferred_interval)
    
    best_df = None
    best_interval = None
    best_score = -1  # Higher is better
    
    for interval in intervals:
        df = data_manager.load_data(symbol, interval)
        if df is None or df.empty:
            continue
        
        # Calculate a cleanliness score (0-100)
        score = 100
        
        # Penalize for NaT timestamps
        nat_pct = df['timestamp'].isna().mean() * 100
        score -= nat_pct
        
        # Penalize for string numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Check for problematic values
                    invalid_pct = df[col].str.contains('USD', na=False, regex=True).mean() * 100
                    score -= invalid_pct
                    
                    # Penalize for non-numeric values
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        score -= 10
        
        # Prefer datasets with more rows (after cleaning)
        # Clean the data
        try:
            df_clean = df[~pd.isna(df['timestamp'])]
            df_clean = df_clean.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df_clean.columns and df_clean[col].dtype == 'object':
                    invalid_mask = df_clean[col].str.contains('USD', na=False, regex=True)
                    df_clean = df_clean[~invalid_mask]
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            df_clean = df_clean.dropna(subset=['timestamp', 'close'])
            
            # Bonus points for more data
            row_bonus = min(20, len(df_clean) / 100)  # Up to 20 points bonus for row count
            score += row_bonus
            
            # If we're at a perfect score or very close, just use this dataset
            if score >= 95:
                return df_clean, interval
                
            # Track the best dataset
            if score > best_score:
                best_score = score
                best_df = df_clean
                best_interval = interval
                
            print(f"Interval {interval}: score {score:.1f}, {len(df_clean)} clean rows")
                
        except Exception as e:
            print(f"Error evaluating {interval} data: {str(e)}")
            score -= 50  # Heavy penalty for errors
    
    # If we found a good dataset
    if best_score > 50 and best_df is not None:
        print(f"Selected {best_interval} data (score: {best_score:.1f})")
        return best_df, best_interval
    
    # If all datasets were problematic, create a simulated one
    print("All datasets had significant issues. Creating simulation data.")
    simulator = DataSimulator()
    return simulator.generate_historical_data(), "simulated"

def run_with_options(use_stored_only=True, download_new=False, list_only=False, use_best_data=False):
    """
    Run the main functionality with options for data sourcing
    
    Args:
        use_stored_only (bool): Whether to only use stored data
        download_new (bool): Whether to download new data regardless of what's stored
        list_only (bool): Whether to only list available data without analysis
        use_best_data (bool): Whether to identify and use the cleanest data source
    """
    if download_new:
        print("Downloading new data from Yahoo Finance...")
        download_and_store_historical_data()
        use_stored_only = False  # After downloading, we can use the fresh data
    
    # Always list available data first
    list_available_data()
    
    # If list_only is True, exit here
    if list_only:
        print("\nListing mode only - not running analysis.")
        return
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    intervals = ['1h', '4h', '1d']
    
    # Print the mode we're running in
    if use_best_data:
        print("\nRunning in BEST DATA mode")
        print("Will automatically select the cleanest data source available.")
    elif use_stored_only:
        print("\nRunning in STORED DATA ONLY mode")
        print("No new data will be downloaded. Using only existing stored data.")
    else:
        print("\nRunning in MIXED mode")
        print("Will attempt to download data if not found in storage.")
    
    # Process each symbol
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print(f"{'='*50}")
        
        if use_best_data:
            # Find the cleanest dataset
            df, interval = find_cleanest_dataset(symbol)
            if df is not None:
                print(f"\nUsing {interval} data for analysis and plotting")
                
                # Debug information
                print(f"Data shape: {df.shape}")
                print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                
                # Add technical indicators for prediction
                engineer = FeatureEngineer()
                df = engineer.add_technical_indicators(df)
                
                # Prepare data for prediction
                preprocessor = DataPreprocessor()
                X, y = preprocessor.prepare_data(df)
                
                if len(X) > 0:
                    # Build and train model
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Build the model
                    model = ModelBuilder.build_lstm_model(
                        sequence_length=60,
                        n_features=X.shape[2]
                    )
                    
                    # Train the model
                    print("Training prediction model...")
                    
                    # Check for NaN or infinite values in training data
                    if np.isnan(X_train).any() or np.isinf(X_train).any():
                        print("Warning: Training data contains NaN or infinite values. Cleaning data...")
                        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    if np.isnan(y_train).any() or np.isinf(y_train).any():
                        print("Warning: Target data contains NaN or infinite values. Cleaning data...")
                        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                    # Use a lower learning rate and add gradient clipping
                    from tensorflow.keras.callbacks import ReduceLROnPlateau
                    from tensorflow.keras.optimizers import Adam
                    
                    # Compile model with lower learning rate and gradient clipping
                    model.compile(
                        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                        loss='mse'
                    )
                    
                    # Add learning rate reduction callback
                    reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=3,
                        min_lr=0.00001,
                        verbose=1
                    )
                        
                    history = model.fit(
                        X_train, y_train,
                        epochs=20,  # Reduced for faster execution
                        batch_size=32,
                        validation_split=0.1,
                        callbacks=[reduce_lr],
                        verbose=1
                    )
                    
                    # Make predictions
                    predictor = PricePredictor(model, preprocessor)
                    
                    # Predict prices for test data
                    predicted_prices = []
                    for i in range(len(X_test)):
                        predicted_price = predictor.predict_next_price(X_test[i:i+1])
                        predicted_prices.append(predicted_price)
                    
                    # Create prediction dataframe
                    pred_indices = range(train_size + 60, train_size + 60 + len(predicted_prices))
                    pred_dates = df['timestamp'].iloc[pred_indices]
                    
                    # Make future predictions (next 30 points)
                    future_predictions = []
                    last_sequence = X[-1:]
                    
                    print("Generating future predictions...")
                    for _ in range(30):
                        # Predict next price
                        next_pred = predictor.predict_next_price(last_sequence)
                        future_predictions.append(next_pred)
                        
                        # Update sequence for next prediction (simple approach)
                        # Create a copy of the last sequence
                        new_seq = last_sequence[0].copy()
                        # Shift the sequence by one step
                        new_seq[:-1] = new_seq[1:]
                        # Add the new prediction as the last value in the sequence (for the 'close' price feature)
                        new_seq[-1, 0] = y[-1]  # Use the scaled version of the prediction
                        
                        # Update last_sequence
                        last_sequence = np.array([new_seq])
                    
                    # Generate future dates
                    last_date = df['timestamp'].iloc[-1]
                    
                    # Determine the time interval in hours
                    interval_hours = 1
                    if interval == '4h':
                        interval_hours = 4
                    elif interval == '1d':
                        interval_hours = 24
                        
                    future_dates = pd.date_range(
                        start=last_date, 
                        periods=31, 
                        freq=f'{interval_hours}h'
                    )[1:]  # Skip the first one which is the last actual data point
                
                # Create figure and subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
                
                # Plot price data
                ax1.plot(df['timestamp'], df['close'], label='Historical Price', color='blue')
                ax1.set_title(f'{symbol} Price Chart ({interval})')
                ax1.set_ylabel('Price (USD)')
                ax1.grid(True)
                
                # Add predictions to the plot if available
                if len(X) > 0:
                    # Plot test predictions in red
                    ax1.plot(pred_dates, predicted_prices, label='Model Prediction', color='red', linestyle='--')
                    
                    # Plot future predictions in red
                    ax1.plot(future_dates, future_predictions, label='Future Prediction', color='red')
                    
                    # Add vertical line to mark where predictions start
                    if len(pred_dates) > 0:
                        ax1.axvline(x=pred_dates.iloc[0], color='gray', linestyle='--', alpha=0.7)
                    
                    # Add vertical line to mark where future predictions start
                    ax1.axvline(x=future_dates[0], color='gray', linestyle='--', alpha=0.7)
                
                ax1.legend()
                
                # Plot volume
                ax2.bar(df['timestamp'], df['volume'], color='green', alpha=0.5)
                ax2.set_ylabel('Volume')
                ax2.grid(True)
                
                # Format x-axis
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Show the plot
                plt.show()
        else:
            # Process each interval
            for interval in intervals:
                print(f"\n{'-'*40}")
                print(f"Processing {symbol} at {interval} interval")
                print(f"{'-'*40}")
                
                # View the data
                view_stored_data(symbol, interval, use_stored_only=use_stored_only)
                
                # Plot the data
                plot_crypto_data(symbol, interval, use_stored_only=use_stored_only)

def list_available_data():
    """List all available cryptocurrency data in storage"""
    data_manager = DataManager()
    
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

if __name__ == "__main__":
    # Choose your data source:
    # 1. Use only stored data without downloading: use_stored_only=True, download_new=False
    # 2. Use stored data first, download if needed: use_stored_only=False, download_new=False
    # 3. Download fresh data and use it: use_stored_only=False, download_new=True
    # 4. Just list what's available: list_only=True
    # 5. Use the best/cleanest data available: use_best_data=True
    
    # Set your preference here:
    use_stored_only = False   # Only use what's already saved
    download_new = True     # Don't download fresh data
    list_only = False        # Run analysis, don't just list data
    use_best_data = True     # Use the cleanest data source available
    
    # Run with the selected options
    run_with_options(use_stored_only, download_new, list_only, use_best_data)