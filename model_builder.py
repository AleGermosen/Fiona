# model_builder.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

class ModelBuilder:
    @staticmethod
    def build_lstm_model(sequence_length, n_features):
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            LSTM(units=50, return_sequences=True,
                 kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True,
                 kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6)),
            Dropout(0.2),
            LSTM(units=50,
                 kernel_regularizer=l2(1e-6), recurrent_regularizer=l2(1e-6)),
            Dropout(0.2),
            Dense(units=1, kernel_regularizer=l2(1e-6))
        ])
        return model