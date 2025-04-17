# feature_engineer.py
import numpy as np
import pandas as pd

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the dataset"""
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure timestamp is properly sorted
        if 'timestamp' in df_copy.columns:
            df_copy = df_copy.sort_values('timestamp')
        
        # Check if we have enough data for indicators
        if len(df_copy) < 30:
            print("Warning: Not enough data for all technical indicators. Padding data...")
            # If very little data, we'll use a simpler set of indicators
            if len(df_copy) < 14:
                print("Warning: Data too short for standard indicators. Using simplified features.")
                # Just add basic stats
                df_copy['MA7'] = df_copy['close'].rolling(window=min(7, len(df_copy))).mean()
                df_copy['MA14'] = df_copy['close']  # Just use close price as placeholder
                df_copy['MA30'] = df_copy['close']  # Just use close price as placeholder
                df_copy['RSI'] = 50  # Neutral RSI
                df_copy['MACD'] = 0
                df_copy['Signal_Line'] = 0
                df_copy['Volatility'] = df_copy['close'].std()
                
                # Fill NaN values
                for col in ['MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']:
                    df_copy[col] = df_copy[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                return df_copy
        
        # Moving averages
        df_copy['MA7'] = df_copy['close'].rolling(window=7).mean()
        df_copy['MA14'] = df_copy['close'].rolling(window=14).mean()
        df_copy['MA30'] = df_copy['close'].rolling(window=30).mean()
        
        # RSI
        delta = df_copy['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Handle division by zero
        loss = loss.replace(0, np.finfo(float).eps)  # Replace zeros with a tiny number
        rs = gain / loss
        df_copy['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df_copy['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_copy['close'].ewm(span=26, adjust=False).mean()
        df_copy['MACD'] = exp1 - exp2
        df_copy['Signal_Line'] = df_copy['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df_copy['Volatility'] = df_copy['close'].rolling(window=14).std()
        
        # Fill NaN values
        for col in ['MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']:
            # First try interpolation
            df_copy[col] = df_copy[col].interpolate(method='linear', limit_direction='both')
            # Then use forward-fill and backward-fill for any remaining NaNs
            df_copy[col] = df_copy[col].ffill().ffill()
            # If still have NaNs (e.g., at the start), fill with reasonable defaults
            if col == 'RSI':
                df_copy[col] = df_copy[col].fillna(50)  # Neutral RSI
            elif col in ['MACD', 'Signal_Line']:
                df_copy[col] = df_copy[col].fillna(0)  # Neutral MACD
            elif 'MA' in col:
                df_copy[col] = df_copy[col].fillna(df_copy['close'])  # Use close price
            else:
                df_copy[col] = df_copy[col].fillna(0)
        
        # Check for infinite values and replace them
        for col in ['MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']:
            if np.isinf(df_copy[col]).any():
                print(f"Warning: Infinite values found in {col}. Replacing with reasonable limits.")
                df_copy[col] = df_copy[col].replace([np.inf, -np.inf], [1e6, -1e6])
            
        return df_copy