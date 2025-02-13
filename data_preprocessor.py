# data_preprocessor.py
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def prepare_data(self, df, sequence_length=60):
        """Prepare data for LSTM model"""
        # Select features for training
        features = ['close', 'volume', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD', 'Signal_Line', 'Volatility']
        data = df[features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predicting the 'close' price
            
        return np.array(X), np.array(y)
    
    def inverse_transform(self, data):
        """Inverse transform the scaled data"""
        dummy = np.zeros((len(data), self.scaler.n_features_in_))
        dummy[:, 0] = data
        return self.scaler.inverse_transform(dummy)[:, 0]

