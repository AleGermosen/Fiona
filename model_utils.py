"""
Model Utilities Module

This module handles operations related to machine learning models,
including saving, loading, and training models for cryptocurrency
price prediction.
"""

import os
import numpy as np
import glob
import shutil
from typing import Optional, Dict, Union, List, Any, Tuple
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def save_model(model, symbol: str, interval: str) -> Optional[str]:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        
    Returns:
        Path to saved model or None if saving failed
    """
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{symbol}_{interval}_model.keras")
    
    try:
        # Save the full model (which includes architecture)
        model.save(model_path, include_optimizer=False)
        print(f"Model saved to {model_path}")
        
        # Save weights separately
        weights_path = os.path.join(model_dir, f"{symbol}_{interval}_weights")
        model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")
        
        return model_path
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        print("Continuing without saving model...")
        return None


def load_saved_model(symbol: str, interval: str) -> Optional[Any]:
    """
    Load a previously trained model from disk if it exists.
    
    Args:
        symbol: The trading pair symbol (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        
    Returns:
        Loaded model or None if loading failed
    """
    model_dir = "saved_models"
    model_path = os.path.join(model_dir, f"{symbol}_{interval}_model.keras")
    weights_path = os.path.join(model_dir, f"{symbol}_{interval}_weights")
    
    if os.path.exists(model_path):
        print(f"Loading previously trained model from {model_path}")
        try:
            from tensorflow.keras.optimizers import Adam
            
            model = tf_load_model(model_path, compile=False)
            
            # Recompile the model with the same settings
            model.compile(
                optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                loss='mse'
            )
            
            print("Model loaded and recompiled successfully")
            return model
        except Exception as e:
            print(f"Error loading complete model: {str(e)}")
            
            # Try loading just the weights if they exist
            if os.path.exists(weights_path) or os.path.exists(weights_path + ".index"):
                try:
                    print(f"Attempting to load weights from {weights_path}")
                    # Create a new model with the same architecture
                    from model_builder import ModelBuilder
                    
                    # Use standard dimensions
                    model = ModelBuilder.build_lstm_model(
                        sequence_length=60,
                        n_features=9  # Typical number of features
                    )
                    
                    # Compile the model
                    model.compile(
                        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
                        loss='mse'
                    )
                    
                    # Load the weights
                    model.load_weights(weights_path)
                    print("Model weights loaded successfully")
                    return model
                except Exception as inner_e:
                    print(f"Error loading weights: {str(inner_e)}")
            
            print("Training a new model instead...")
            return None
    else:
        print(f"No saved model found for {symbol} at {interval} interval")
        return None


def delete_saved_models(symbol: Optional[str] = None, interval: Optional[str] = None) -> int:
    """
    Delete saved models to start fresh.
    
    Args:
        symbol: Optional - specific symbol to delete models for
        interval: Optional - specific interval to delete models for
        
    Returns:
        Number of files deleted
    """
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        print("No saved models directory exists.")
        return 0
        
    files_deleted = 0
    
    if symbol and interval:
        # Delete specific model
        model_path = os.path.join(model_dir, f"{symbol}_{interval}_model.keras")
        weights_path = os.path.join(model_dir, f"{symbol}_{interval}_weights")
        
        if os.path.exists(model_path):
            os.remove(model_path)
            files_deleted += 1
            print(f"Deleted {model_path}")
            
        # Delete weights directory or files
        if os.path.isdir(weights_path):
            shutil.rmtree(weights_path)
            files_deleted += 1
            print(f"Deleted weights directory {weights_path}")
        else:
            # Check for TensorFlow format files
            weight_files = glob.glob(weights_path + "*")
            for wf in weight_files:
                os.remove(wf)
                files_deleted += 1
                print(f"Deleted {wf}")
    else:
        # Delete all models
        # Delete all .keras files
        keras_files = glob.glob(os.path.join(model_dir, "*.keras"))
        
        for file in keras_files:
            os.remove(file)
            files_deleted += 1
            print(f"Deleted {file}")
        
        # Delete all weight directories
        weight_dirs = [d for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d)) and d.endswith('_weights')]
        
        for wd in weight_dirs:
            shutil.rmtree(os.path.join(model_dir, wd))
            files_deleted += 1
            print(f"Deleted weights directory {wd}")
        
        # Delete all other weight files
        weight_files = glob.glob(os.path.join(model_dir, "*_weights.*"))
        for wf in weight_files:
            os.remove(wf)
            files_deleted += 1
            print(f"Deleted {wf}")
    
    if files_deleted == 0:
        print(f"No saved models found{' for ' + symbol + ' ' + interval if symbol and interval else ''}")
    else:
        print(f"Deleted {files_deleted} model files/directories")
        
    return files_deleted


def get_training_callbacks() -> List[Any]:
    """
    Get standard training callbacks for model training.
    
    Returns:
        List of Keras callbacks for training
    """
    # Add learning rate reduction callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
    
    return [reduce_lr]


def train_model(model, X_train, y_train, epochs: int = 20, batch_size: int = 32, 
               validation_split: float = 0.2) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Train a model with standard parameters and error handling.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (trained_model, history_dict)
    """
    # Clean input data
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("Warning: Training data contains NaN or infinite values. Cleaning data...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
        
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        print("Warning: Target data contains NaN or infinite values. Cleaning data...")
        y_train = np.nan_to_num(y_train, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Get training callbacks
    callbacks = get_training_callbacks()
    
    # Train the model
    print(f"Training model with {len(X_train)} samples for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history.history


def get_or_train_model(X, X_train, y_train, symbol: str, interval: str, 
                     use_saved_model: bool = True, use_pretrained_only: bool = False) -> Optional[Any]:
    """
    Get a saved model or train a new one.
    
    Args:
        X: Full feature set (for shape information)
        X_train: Training features
        y_train: Training targets
        symbol: The trading pair symbol
        interval: Time interval
        use_saved_model: Whether to try loading a saved model
        use_pretrained_only: Whether to only use pre-trained models and not train new ones
        
    Returns:
        Trained model or None if failed
    """
    # Try to load a saved model if requested
    model = None
    if use_saved_model:
        model = load_saved_model(symbol, interval)
    
    # If no saved model available or not using saved models, train a new one
    if model is None:
        if use_pretrained_only:
            print(f"No pre-trained model available for {symbol} at {interval} interval and use_pretrained_only is enabled.")
            print(f"Skipping {interval} interval. Run pretrain_models.py to train models for all intervals.")
            return None
            
        print("Creating and training a new model...")
        
        from model_builder import ModelBuilder
        
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
        model, _ = train_model(model, X_train, y_train)
        
        # Save the trained model
        # save_model(model, symbol, interval)
        model.save("saved_models/{}_{}_model.keras".format(symbol, interval))
    else:
        print("Using previously trained model for predictions")
    
    return model


def generate_future_predictions(predictor, preprocessor, X, df, 
                               interval: str, future_count: int = 90) -> Tuple[List[float], Any]:
    """
    Generate future price predictions beyond available data.
    
    Args:
        predictor: PricePredictor instance
        preprocessor: DataPreprocessor instance
        X: Feature matrix
        df: DataFrame with historical data
        interval: Time interval (e.g., '1h', '4h', '1d')
        future_count: Number of future predictions to generate
        
    Returns:
        Tuple of (future_predictions, future_dates)
    """
    import pandas as pd
    import numpy as np
    import logging
    from feature_engineer import FeatureEngineer
    
    logger = logging.getLogger("crypto_prediction")
    logger.info(f"Generating {future_count} future predictions for {interval} timeframe...")
    
    # Generate future dates first
    last_date = df['timestamp'].iloc[-1]
    
    # Determine the time interval in hours
    interval_hours = 1
    if interval.startswith("combined_"):
        # For combined datasets, use the base interval (after "combined_")
        base_interval = interval.split("_")[1]
        if base_interval == '4h':
            interval_hours = 4
        elif base_interval == '1d':
            interval_hours = 24
    elif interval == '4h':
        interval_hours = 4
    elif interval == '1d':
        interval_hours = 24
        
    future_dates = pd.date_range(
        start=last_date, 
        periods=future_count+1, 
        freq=f'{interval_hours}h'
    )[1:]  # Skip the first one which is the last actual data point
    
    # Create a copy of the dataframe with only the necessary columns
    future_df = df.copy()
    
    # Initialize list for future predictions
    future_predictions = []
    
    # Get the current sequence for the first prediction
    last_sequence = X[-1:].copy()
    
    # Create feature engineer for updating technical indicators
    feature_engineer = FeatureEngineer()
    
    # Generate one prediction per future date/timeframe
    for i, future_date in enumerate(future_dates):
        # Predict next price using the current sequence
        next_pred = predictor.predict_next_price(last_sequence)
        future_predictions.append(next_pred)
        
        # Log the prediction with timeframe information
        logger.info(f"Predicted price for {future_date.strftime('%Y-%m-%d %H:%M:%S')}: ${next_pred:.2f}")
        
        # Only update the sequence if we have more predictions to make
        if i < len(future_dates) - 1:
            # Add the predicted price to our future dataframe
            new_row = pd.DataFrame({
                'timestamp': [future_date],
                'open': [next_pred],  # Using predicted close as open
                'high': [next_pred * 1.001],  # Slight adjustment for simulation
                'low': [next_pred * 0.999],   # Slight adjustment for simulation
                'close': [next_pred],
                'volume': [future_df['volume'].tail(30).mean()]  # Use avg of recent volume
            })
            
            # Append to the dataframe
            future_df = pd.concat([future_df, new_row], ignore_index=True)
            
            # Recalculate all technical indicators with the new price data
            future_df = feature_engineer.add_technical_indicators(future_df)
            
            # Extract the latest row with all indicators
            latest_features = future_df.iloc[-1][['close', 'volume', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']].values
            
            # Scale the new features (match how the original features were processed)
            dummy = np.zeros((1, len(latest_features)))
            dummy[0, :] = latest_features
            scaled_features = preprocessor.scaler.transform(dummy)[0]
            
            # Update the sequence for the next iteration
            new_seq = last_sequence[0].copy()
            # Shift the values (remove oldest, make room for newest)
            new_seq[:-1] = new_seq[1:]
            # Add the new features as the last element in the sequence
            new_seq[-1, :] = scaled_features
            
            # Update the sequence for the next prediction
            last_sequence = np.array([new_seq])
    
    return future_predictions, future_dates 