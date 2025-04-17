"""
Cryptocurrency Price Prediction Application

This is the main entry point for the cryptocurrency price prediction application.
It provides functionality to download, analyze, and predict cryptocurrency prices
using historical data and machine learning models.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Tuple, Any

# Import our modules
from data_downloader import download_from_coingecko, download_and_store_historical_data
from data_utils import (
    find_cleanest_dataset, list_available_data, view_stored_data, 
    prepare_data_for_modeling, clean_dataframe, join_timeframe_datasets
)
from model_utils import (
    get_or_train_model, delete_saved_models,
    generate_future_predictions, save_model, load_saved_model, train_model
)
from visualization import plot_crypto_data_with_predictions

# Import other components
from data_collector import DataCollector
from feature_engineer import FeatureEngineer
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder
from predictor import PricePredictor
from data_simulator import DataSimulator
from data_manager import DataManager
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Global variables
VALIDATION_SPLIT = 0.2
FUTURE_COUNT = 730

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crypto_prediction")

# Load environment variables from .env file
load_dotenv()


def pretrain_models_for_all_intervals(data_manager: DataManager, 
                                    symbols: List[str] = ['BTCUSDT'], 
                                    intervals: List[str] = ['1h', '4h', '1d']) -> int:
    """
    Pre-train models for all specified symbols and intervals.
    
    Args:
        data_manager: DataManager instance
        symbols: List of trading pair symbols to train models for
        intervals: List of intervals to train models for
        
    Returns:
        Number of successfully trained models
    """
    logger.info("Starting pre-training for all intervals")
    
    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Check if we have data, if not, download it
    if not data_manager.get_available_symbols():
        logger.info("No data found. Downloading historical data...")
        download_and_store_historical_data(data_manager)
    
    # Track success count
    success_count = 0
    total_count = len(symbols) * len(intervals)
    
    for symbol in symbols:
        for interval in intervals:
            logger.info(f"Processing {symbol} at {interval} interval")
            
            # Check if model already exists
            existing_model = load_saved_model(symbol, interval)
            if existing_model is not None:
                logger.info(f"Model for {symbol} at {interval} already exists. Skipping.")
                success_count += 1
                continue
            
            # View the data
            df = view_stored_data(
                data_manager, 
                symbol, 
                interval, 
                rows=5, 
                include_stats=True,
                use_stored_only=True
            )
            
            if df is None or len(df) < 100:  # Need minimum data for meaningful training
                logger.warning(f"Insufficient data for {symbol} at {interval} interval. Skipping.")
                continue
            
            # Add technical indicators
            engineer = FeatureEngineer()
            df = engineer.add_technical_indicators(df)
            
            # Prepare data for prediction
            preprocessor = DataPreprocessor()
            X, y = preprocessor.prepare_data(df)
            
            if len(X) == 0:
                logger.error("Insufficient data for modeling after preparation")
                continue
            
            # Split into training and test sets
            train_size = int(len(X) * (1 - VALIDATION_SPLIT))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Clean input data
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logger.warning("Training data contains NaN or infinite values. Cleaning data...")
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if np.isnan(y_train).any() or np.isinf(y_train).any():
                logger.warning("Target data contains NaN or infinite values. Cleaning data...")
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Build the model
            model = ModelBuilder.build_lstm_model(
                sequence_length=60,
                n_features=X.shape[2]
            )
            
            # Compile model with lower learning rate and gradient clipping
            model.compile(
                optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                loss='mse'
            )
            
            # Train the model
            logger.info(f"Starting training for {symbol} at {interval} with {len(X_train)} samples...")
            model, _ = train_model(model, X_train, y_train)
            
            # Save the trained model
            model_path = save_model(model, symbol, interval)
            if model_path:
                logger.info(f"Successfully trained and saved model for {symbol} at {interval} to {model_path}")
                success_count += 1
            else:
                logger.error(f"Failed to save model for {symbol} at {interval}")
    
    # Report summary
    logger.info(f"Pre-training complete. Successfully trained {success_count} out of {total_count} models.")
    return success_count


def process_with_best_data(data_manager: DataManager, symbol: str, use_saved_model: bool) -> None:
    """
    Process cryptocurrency data using the best available dataset.
    
    Args:
        data_manager: DataManager instance
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        use_saved_model: Whether to use previously saved models
        use_pretrained_only: Whether to only use pre-trained models and skip intervals without models
    """
    # Find the cleanest dataset
    df, interval = find_cleanest_dataset(data_manager, symbol, preferred_interval='1h')
    
    if df is None or interval is None:
        logger.error(f"No suitable data found for {symbol}")
        return
        
    logger.info(f"Using {interval} data for analysis and plotting")
    
    # Debug information
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Filter data to start from August 2024
    # start_date = pd.to_datetime('2024-08-01')
    # df = df[df['timestamp'] >= start_date].reset_index(drop=True)
    # logger.info(f"After filtering for dates >= August 2024: {len(df)} rows")
    
    # Add technical indicators
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    
    # Prepare data for prediction
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    if len(X) == 0:
        logger.error("Insufficient data for modeling after preparation")
        return
    
    # Split into training and test sets
    train_size = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Get or train a model
    model = get_or_train_model(X, X_train, y_train, symbol, interval, use_saved_model)
    
    if model is None:
        logger.error("Failed to get or train a model")
        return
    
    # Make predictions
    predictor = PricePredictor(model, preprocessor)
    
    # Generate test predictions
    predicted_prices = [predictor.predict_next_price(X_test[i:i+1]) 
                      for i in range(len(X_test))]
    
    # Create prediction dataframe
    pred_indices = np.arange(train_size + 60, train_size + 60 + len(predicted_prices))
    pred_dates = df['timestamp'].iloc[pred_indices]
    
    # Generate future predictions
    future_predictions, future_dates = generate_future_predictions(
        predictor, preprocessor, X, df, interval, FUTURE_COUNT
    )
    
    # Plot the data with predictions
    plot_crypto_data_with_predictions(
        df=df,
        symbol=symbol, 
        interval=interval,
        pred_dates=pred_dates,
        predicted_prices=predicted_prices,
        future_dates=future_dates,
        future_predictions=future_predictions
    )


def process_with_intervals(data_manager: DataManager, symbol: str, intervals: List[str], 
                         use_stored_only: bool, use_saved_model: bool) -> None:
    """
    Process cryptocurrency data using specified intervals.
    
    Args:
        data_manager: DataManager instance
        symbol: Trading pair symbol (e.g., 'BTCUSDT') 
        intervals: List of intervals to process (e.g., ['1h', '4h'])
        use_stored_only: Whether to only use stored data
        use_saved_model: Whether to use previously saved models
        use_pretrained_only: Whether to only use pre-trained models and skip intervals without models
    """
    for interval in intervals:
        logger.info(f"Processing {symbol} at {interval} interval")
        
        # View the data
        df = view_stored_data(
            data_manager, 
            symbol, 
            interval, 
            rows=5, 
            include_stats=True,
            use_stored_only=use_stored_only
        )
        
        if df is None:
            logger.warning(f"No data available for {symbol} at {interval} interval")
            continue
        
        # Add technical indicators
        engineer = FeatureEngineer()
        df = engineer.add_technical_indicators(df)
        
        # Prepare data for prediction
        preprocessor = DataPreprocessor()
        X, y = preprocessor.prepare_data(df)
        
        if len(X) == 0:
            logger.error("Insufficient data for modeling after preparation")
            continue
        
        # Split into training and test sets
        train_size = int(len(X) * (1 - VALIDATION_SPLIT))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Get or train a model
        model = get_or_train_model(X, X_train, y_train, symbol, interval, use_saved_model)
        
        if model is None:
            logger.error("Failed to get or train a model")
            continue
        
        # Make predictions
        predictor = PricePredictor(model, preprocessor)
        
        # Generate test predictions
        predicted_prices = [predictor.predict_next_price(X_test[i:i+1]) 
                          for i in range(len(X_test))]
        
        # Create prediction dataframe
        pred_indices = np.arange(train_size + 60, train_size + 60 + len(predicted_prices))
        pred_dates = df['timestamp'].iloc[pred_indices]
        
        # Generate future predictions
        future_predictions, future_dates = generate_future_predictions(
            predictor, preprocessor, X, df, interval, FUTURE_COUNT
        )
        
        # Plot the data with predictions
        plot_crypto_data_with_predictions(
            df=df,
            symbol=symbol, 
            interval=interval,
            pred_dates=pred_dates,
            predicted_prices=predicted_prices,
            future_dates=future_dates,
            future_predictions=future_predictions
        )


def process_combined_dataset(df: pd.DataFrame, symbol: str, interval: str, 
                           use_saved_model: bool) -> None:
    """
    Process a combined dataset from multiple timeframes for model training.
    
    Args:
        df: DataFrame containing combined data from multiple timeframes
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval to use for model naming (e.g., '1h')
        use_saved_model: Whether to use previously saved models
        use_pretrained_only: Whether to only use pre-trained models
    """
    if df is None or df.empty:
        logger.error(f"No data available for processing")
        return
        
    logger.info(f"Processing combined timeframe data")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Add technical indicators
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)
    
    # Prepare data for prediction
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(df)
    
    if len(X) == 0:
        logger.error("Insufficient data for modeling after preparation")
        return
    
    # Split into training and test sets
    train_size = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Model name will use the provided interval (typically '1h')
    model_name = f"combined_{interval}"
    
    # Get or train a model
    model = get_or_train_model(X, X_train, y_train, symbol, model_name, use_saved_model)
    
    if model is None:
        logger.error(f"Failed to get or train model for {symbol} with combined data")
        return
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    
    logger.info(f"Model Evaluation for combined data:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    
    # Make prediction for next price
    predictor = PricePredictor(model, preprocessor)
    last_sequence = X_test[-1:]
    next_price = predictor.predict_next_price(last_sequence)
    
    logger.info(f"Latest price: ${df['close'].iloc[-1]:.2f}")
    logger.info(f"Predicted next price: ${next_price:.2f}")
    
    # Calculate percent change
    price_change = ((next_price - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
    logger.info(f"Predicted price change: {price_change:.2f}%")
    
    # Generate test predictions
    predicted_prices = [predictor.predict_next_price(X_test[i:i+1]) 
                      for i in range(len(X_test))]
    
    # Create prediction dataframe indices
    pred_indices = np.arange(train_size + 60, train_size + 60 + len(predicted_prices))
    pred_dates = df['timestamp'].iloc[pred_indices]
    
    # Generate forecast for future predictions
    future_predictions, future_dates = generate_future_predictions(
        predictor, preprocessor, X, df, interval, FUTURE_COUNT
    )
    
    # Plot the data with predictions
    plot_crypto_data_with_predictions(
        df=df,
        symbol=symbol, 
        interval=f"combined_{interval}",
        pred_dates=pred_dates,
        predicted_prices=predicted_prices,
        future_dates=future_dates,
        future_predictions=future_predictions
    )
    
    # Generate forecast
    forecast_df = predictor.forecast_future_prices(df, steps=24)
    logger.info(f"24-hour forecast generated")


def run_with_options(use_stored_only: bool = True, 
                   download_new: bool = False, 
                   list_only: bool = False, 
                   use_best_data: bool = True, 
                   use_saved_model: bool = True, 
                   reset_models: bool = False,
                   use_combined_timeframes: bool = False) -> None:
    """
    Run the main functionality with options for data sourcing.
    
    Args:
        use_stored_only: Whether to only use stored data
        download_new: Whether to download new data regardless of what's stored
        list_only: Whether to only list available data without analysis
        use_best_data: Whether to identify and use the cleanest data source
        use_saved_model: Whether to use previously saved models
        reset_models: Whether to delete saved models and start fresh
        use_pretrained_only: Whether to only use pre-trained models and skip intervals without models
        pretrain_all: Whether to pre-train models for all intervals before analysis
        use_combined_timeframes: Whether to join different timeframe datasets before training
    """
    data_manager = DataManager()
    
    # Handle model reset if requested
    if reset_models:
        logger.info("Deleting all saved models...")
        delete_saved_models()
    
    # Handle data download if requested
    if download_new:
        logger.info("Downloading new data from CoinGecko...")
        download_and_store_historical_data(data_manager)
        use_stored_only = False  # After downloading, we can use the fresh data
    
    # List available data
    list_available_data(data_manager)
    
    # Exit if only listing was requested
    if list_only:
        logger.info("Listing mode only - not running analysis.")
        return
    
    # Configure runtime settings
    symbols = ['BTCUSDT']  # Could be expanded to ['BTCUSDT', 'ETHUSDT'] etc.
    intervals = ['1h', '4h', '1d']     # Could be expanded to ['1h', '4h', '1d'] etc.
    
    # Print operation mode
    _print_operation_mode(use_best_data, use_stored_only)
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        
        # If using combined timeframes, process with joined data
        if use_combined_timeframes:
            logger.info("Using combined timeframe data for model training")
            from data_utils import join_timeframe_datasets
            
            combined_df = join_timeframe_datasets(data_manager, symbol, intervals)
            if combined_df is not None:
                logger.info(f"Combined dataset created with {len(combined_df)} records")
                
                # Use interval '1h' for model name since that's the most granular
                process_combined_dataset(combined_df, symbol, '4h', use_saved_model)
            else:
                logger.error("Failed to create combined dataset. No data available.")
        elif use_best_data:
            process_with_best_data(data_manager, symbol, use_saved_model)
        else:
            process_with_intervals(data_manager, symbol, intervals, use_stored_only, use_saved_model)


def _print_operation_mode(use_best_data: bool, use_stored_only: bool) -> None:
    """Print the current operation mode to console."""
    if use_best_data:
        logger.info("Running in BEST DATA mode")
        logger.info("Will automatically select the cleanest data source available.")
    elif use_stored_only:
        logger.info("Running in STORED DATA ONLY mode")
        logger.info("No new data will be downloaded. Using only existing stored data.")
    else:
        logger.info("Running in MIXED mode")
        logger.info("Will attempt to download data if not found in storage.")


def main(use_simulation: bool = False) -> None:
    """
    Main entry point for quick testing or direct execution.
    
    Args:
        use_simulation: Whether to use simulated data
    """
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
        # Try to get data from CoinGecko
        df = download_from_coingecko(symbol="BTC", days=90, interval="hourly")
        df = download_from_coingecko(symbol="BTC", days=365, interval="daily")
        # Fallback to simulation if CoinGecko fails
        if df is None:
            logger.warning("Falling back to simulated data...")
            simulator = DataSimulator()
            df = simulator.generate_historical_data()
    
    df = engineer.add_technical_indicators(df)
    X, y = preprocessor.prepare_data(df)
    
    train_size = int(len(X) * (1 - VALIDATION_SPLIT))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = ModelBuilder.build_lstm_model(
        sequence_length=60,
        n_features=X.shape[2]
    )
    
    # Check for NaN or infinite values in training data
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        logger.warning("Training data contains NaN or infinite values. Cleaning data...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        logger.warning("Target data contains NaN or infinite values. Cleaning data...")
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compile model
    from tensorflow.keras.optimizers import Adam
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='mse'
    )
    
    # Train model
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
        
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=VALIDATION_SPLIT,
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Make prediction
    predictor = PricePredictor(model, preprocessor)
    last_sequence = X_test[-1:]
    next_price = predictor.predict_next_price(last_sequence)
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"DEBUG [{current_time}] - Predicted next price: ${next_price:.2f}")