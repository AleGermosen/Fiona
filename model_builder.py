# model_builder.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    @staticmethod
    def build_lstm_model(sequence_length, n_features):
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, n_features)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model