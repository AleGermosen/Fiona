# data_preprocessor.py
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import numpy as np
import pandas as pd

class DataPreprocessor:
    def __init__(self):
        # Use RobustScaler instead of MinMaxScaler to better handle outliers
        self.scaler = RobustScaler()
    
    def prepare_data(self, df, sequence_length=60):
        """Prepare data for LSTM model"""
        # Select features for training
        features = ['close', 'volume', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check for necessary features
        missing_features = [feat for feat in features if feat not in df_copy.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Use only available features
            features = [feat for feat in features if feat in df_copy.columns]
            
            if not features:
                raise ValueError("No valid features found for model training")
        
        # Replace infinity values with NaN
        for col in features:
            if col in df_copy.columns:
                df_copy[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Check for NaN values and fill them
        if df_copy[features].isna().any().any():
            print("Warning: NaN values found in features. Filling with interpolation and forward/backward fill.")
            # Try to interpolate first
            df_copy[features] = df_copy[features].interpolate(method='linear', limit_direction='both')
            # If still have NaNs, use forward fill and backward fill
            df_copy[features] = df_copy[features].fillna(method='ffill').fillna(method='bfill')
            # If still have NaNs, fill with zeros
            df_copy[features] = df_copy[features].fillna(0)
        
        # Get feature data
        data = df_copy[features].values
        
        # Additional check for any invalid values
        if np.isnan(data).any() or np.isinf(data).any():
            print("Warning: Data contains NaN or infinite values after preprocessing. Using np.nan_to_num...")
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale the data with outlier-robust scaling
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predicting the 'close' price
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Final check for any remaining NaN or inf values
        if np.isnan(X_array).any() or np.isinf(X_array).any():
            print("Warning: Sequences contain NaN or infinite values. Cleaning sequences...")
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if np.isnan(y_array).any() or np.isinf(y_array).any():
            print("Warning: Target values contain NaN or infinite values. Cleaning targets...")
            y_array = np.nan_to_num(y_array, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return X_array, y_array
    
    def inverse_transform(self, data):
        """Inverse transform the scaled data"""
        # Create dummy array with zeros for all features
        dummy = np.zeros((len(data), self.scaler.n_features_in_))
        # Put the predicted values in the first column (close price)
        dummy[:, 0] = data
        # Inverse transform and convert to dense array if needed
        result = self.scaler.inverse_transform(dummy)
        # Debug print to check values
        if len(data) < 10:  # Only print for small arrays to avoid console spam
            original_values = np.asarray(result)[:, 0]
            print(f"DEBUG - Inverse transformed values: min={min(original_values):.2f}, max={max(original_values):.2f}, sample={original_values[:3]}")
        return np.asarray(result)[:, 0]

