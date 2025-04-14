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

## Data Source Migration: From Binance to Yahoo Finance

### Overview

Due to issues with accessing the Binance API, the project has been updated to use Yahoo Finance as an alternative data source for cryptocurrency price data. This document outlines the changes made and how to use the updated functionality.

### Key Changes

1. **New Data Source**: Added `download_from_yahoo()` function to retrieve data from Yahoo Finance API
2. **Symbol Format**: Added automatic conversion between Binance symbol format (e.g., 'BTCUSDT') and Yahoo Finance format (e.g., 'BTC-USD')
3. **Fallback Mechanism**: Implemented a fallback to simulated data if Yahoo Finance data is not available
4. **Updated Functions**: Modified `download_and_store_historical_data()` and `simulate_with_real_data()` to use Yahoo Finance

### How to Use

#### Prerequisites

Install the Yahoo Finance API:
```
pip install yfinance
```

#### Downloading Data

```python
# Download Bitcoin data for the last 6 months at 1-hour intervals
df = download_from_yahoo(symbol="BTC-USD", period="6mo", interval="1h")
```

#### Valid Symbol Formats
When using the `download_from_yahoo()` function:
- Use Yahoo Finance format directly: `BTC-USD`, `ETH-USD`
- Or use Binance format which will be converted: `BTCUSDT` → `BTC-USD`

#### Available Periods
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

#### Available Intervals
- `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

### Error Handling

The Yahoo Finance integration includes robust error handling:
- Empty dataset detection and handling
- Exception catching for network issues
- Automatic fallback to simulation when data is unavailable
- Proper symbol format validation and conversion
- JSON serialization error prevention
- Three-tier fallback strategy:
  1. Try to load from local storage
  2. If not available, try Yahoo Finance
  3. If Yahoo Finance fails, use simulation

Additional error resilience features:
- Data type standardization before storage
- Metadata serialization protection
- Graceful handling of missing or malformed data
- Automatic retry mechanisms when appropriate

### Visualization

The project now includes a plotting function to visualize cryptocurrency price data:

```python
# Plot Bitcoin price data
plot_crypto_data('BTCUSDT', '1h')
```

The plotting system is also robust against data issues:
- Automatically attempts to retrieve data if not locally available
- Falls back to downloading if local storage is empty
- Can generate simulated data if all other sources fail
- Handles date filtering gracefully

This creates a dual-panel chart showing:
1. Price movement over time
2. Volume information

### Data Flow

1. Try to load data from local storage
2. If not available, download from Yahoo Finance
3. If download fails, fall back to simulated data
4. Process and model the data regardless of source

### Data Source Control Options

The project now includes options to control where data comes from:

#### Running Options

In `main.py`, you can set the following flags:

```python
# 1. Use only stored data without downloading
use_stored_only = True
download_new = False

# 2. Use stored data first, download if not available
use_stored_only = False
download_new = False

# 3. Download fresh data and use it
use_stored_only = False
download_new = True
```

#### Command Line Usage

Simply run `main.py` after setting your preferences:

```
python main.py
```

### Timezone Handling

The system now includes robust timezone handling:
- Automatically converts timezone-aware timestamps to timezone-naive format
- Prevents timezone comparison errors when combining data from different sources
- Maintains consistent timestamp formatting across all data storage and processing

This ensures that data from Yahoo Finance (which may include timezone information) can be seamlessly combined with stored data or simulation data.

### Notes

- Yahoo Finance has different rate limits than Binance, but they are generally sufficient for this project
- The data format is standardized internally, so all existing model functionality continues to work
- Historical data downloaded from Yahoo Finance is stored in the same format as before for compatibility
- The implementation checks for empty datasets to prevent errors in data processing
