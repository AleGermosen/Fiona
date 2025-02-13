# Cryptocurrency Trading Bot Documentation

## Overview
A machine learning-based cryptocurrency trading bot that uses historical price data to predict future price movements. The system can work with both real Binance data and simulated data for testing purposes.

## System Components

### Data Collection (`data_collector.py`, `data_simulator.py`)
- `BinanceCollector`: Fetches historical data from Binance API
- `DataSimulator`: Generates synthetic market data for testing
- `DataCollector`: Manages data sourcing from either Binance or simulation

### Data Management (`data_manager.py`, `data_storage.py`)
- Handles data persistence and retrieval
- Maintains organized storage structure
- Supports metadata tracking
- Implements data validation and cleanup

### Feature Engineering (`feature_engineer.py`)
Calculates technical indicators:
- Moving averages (7, 14, 30 periods)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Volatility measures

### Data Preprocessing (`data_preprocessor.py`)
- Scales features using MinMaxScaler
- Creates sequence data for LSTM model
- Handles inverse transformation for predictions

### Model Architecture (`model_builder.py`)
LSTM neural network with:
- 3 LSTM layers (50 units each)
- Dropout layers (0.2)
- Dense output layer
- Adam optimizer

### Price Prediction (`predictor.py`)
- Handles model inference
- Transforms predictions back to original scale

## Setup Requirements

### Dependencies
- Python 3.x
- TensorFlow
- pandas
- numpy
- python-binance
- scikit-learn

### Environment Variables
Required in `.env` file:
```
API_KEY=your_binance_api_key
API_SECRET=your_binance_api_secret
```

## Usage Examples

### Using Real Data
```python
collector = DataCollector(api_key, api_secret)
data = collector.get_historical_data(
    symbol="BTCUSDT",
    interval="1h",
    start_str="1 month ago UTC"
)
```

### Using Simulated Data
```python
collector = DataCollector(use_simulation=True)
data = collector.get_historical_data(
    symbol="BTCUSDT",
    interval="1h",
    start_str="1 month ago UTC"
)
```

### Data Storage
```python
data_manager = DataManager()
data_manager.save_raw_data(df, "BTCUSDT", "1h")
stored_data = data_manager.load_data("BTCUSDT", "1h")
```

## Directory Structure
```
crypto_data/
├── raw/           # Raw market data
├── processed/     # Processed datasets
├── cache/         # Temporary data
└── metadata/      # Data metadata files
```

## Error Handling
- Data validation checks
- Logging system for tracking operations
- Graceful fallbacks for API failures

## Performance Considerations
- Efficient data storage and retrieval
- Caching support
- Cleanup of old data
- Memory-efficient data processing
