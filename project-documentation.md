# Cryptocurrency Trading Bot Documentation

## Overview
A machine learning-based cryptocurrency trading bot that uses historical price data to predict future price movements. The system can work with multiple data sources including CoinGecko, Yahoo Finance, Binance API, and simulated data for testing purposes.

## System Components

### Data Collection (`data_collector.py`, `data_simulator.py`, `data_downloader.py`)
- `BinanceCollector`: Fetches historical data from Binance API
- `DataSimulator`: Generates synthetic market data for testing
- `DataCollector`: Manages data sourcing from either Binance or simulation
- `download_from_coingecko()`: Fetches data from CoinGecko API
- `download_from_yahoo()`: Fetches data from Yahoo Finance API

### Data Management (`data_manager.py`, `data_storage.py`, `data_utils.py`)
- Handles data persistence and retrieval
- Maintains organized storage structure
- Supports metadata tracking
- Implements data validation and cleanup
- Provides utilities for finding the best/cleanest dataset

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

### Model Architecture (`model_builder.py`, `model_utils.py`)
LSTM neural network with:
- 3 LSTM layers (50 units each)
- Dropout layers (0.2)
- Dense output layer
- Adam optimizer
- Model persistence and loading

### Price Prediction (`predictor.py`)
- Handles model inference
- Transforms predictions back to original scale
- Generates future price predictions

### Visualization (`visualization.py`)
- Plot cryptocurrency price history
- Display technical indicators
- Visualize model predictions
- Compare actual vs predicted prices
- Interactive plots for analysis

### Web Application (`app.py`)
- Interactive interface for the trading bot
- Data visualization dashboard
- Model training and prediction controls
- Automated data refresh and processing
- Configurable settings for different trading strategies

## Setup Requirements

### Dependencies
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy
- pycoingecko
- yfinance (optional)
- python-binance (optional)
- scikit-learn
- matplotlib
- plotly

### Environment Variables
Required in `.env` file:
```
API_KEY=your_binance_api_key (optional)
API_SECRET=your_binance_api_secret (optional)
```

## Usage Examples

### Using Real Data from CoinGecko
```python
# Download Bitcoin data for the last 90 days at hourly intervals
df = download_from_coingecko(symbol="BTC", days=90, interval="hourly")
```

### Using the Web Application
```
# Run the web application
python app.py
```

### Processing Multiple Intervals
```python
data_manager = DataManager()
process_with_intervals(
    data_manager=data_manager,
    symbol="BTCUSDT",
    intervals=["1h", "4h"],
    use_stored_only=False,
    use_saved_model=True
)
```

### Using Advanced Visualization
```python
# Visualize price predictions
plot_crypto_data_with_predictions(
    df=df,
    symbol=symbol, 
    interval=interval,
    pred_dates=pred_dates,
    predicted_prices=predicted_prices,
    future_dates=future_dates,
    future_predictions=future_predictions
)
```

## Directory Structure
```
project_root/
├── app.py                # Web application entry point
├── main.py               # CLI entry point
├── data_collector.py     # Data collection from Binance
├── data_downloader.py    # Data downloading from CoinGecko/Yahoo
├── data_simulator.py     # Data simulation utilities
├── data_manager.py       # Data persistence management
├── data_storage.py       # Data storage operations
├── data_utils.py         # Data utility functions
├── data_preprocessor.py  # Data preprocessing
├── feature_engineer.py   # Technical indicator calculation
├── model_builder.py      # Model architecture definition
├── model_utils.py        # Model training and evaluation
├── predictor.py          # Price prediction
├── visualization.py      # Data visualization
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
└── crypto_data/          # Data storage directory
    ├── raw/              # Raw market data
    ├── processed/        # Processed datasets
    ├── cache/            # Temporary data
    └── metadata/         # Data metadata files
└── saved_models/         # Stored trained models
```

## Error Handling
- Data validation checks
- Logging system for tracking operations
- Graceful fallbacks for API failures
- Automatic switch between data sources

## Performance Considerations
- Efficient data storage and retrieval
- Caching support
- Cleanup of old data
- Memory-efficient data processing
- Model checkpointing for resuming training

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

## Data Source Migration: From Yahoo Finance to CoinGecko

### Overview

The project has been updated to use CoinGecko as the data source for cryptocurrency price data. This change improves data reliability and provides access to more cryptocurrency data. This document outlines the changes made and how to use the updated functionality.

### Key Changes

1. **New Data Source**: Added `download_from_coingecko()` function to retrieve data from CoinGecko API
2. **Symbol Format**: Added automatic conversion between Binance symbol format (e.g., 'BTCUSDT') and CoinGecko format (e.g., 'bitcoin')
3. **Fallback Mechanism**: Implemented a fallback to simulated data if CoinGecko data is not available
4. **Updated Functions**: Modified all relevant functions to use CoinGecko instead of Yahoo Finance

### How to Use

#### Prerequisites

Install the CoinGecko API:
```
pip install pycoingecko
```

#### Downloading Data

```python
# Download Bitcoin data for the last 90 days at hourly intervals
df = download_from_coingecko(symbol="BTC", days=90, interval="hourly")
```

#### Valid Symbol Formats
When using the `download_from_coingecko()` function:
- Use common crypto symbols: `BTC`, `ETH`
- Or use Binance format which will be converted: `BTCUSDT` → `BTC`
- Or use Yahoo Finance format which will be converted: `BTC-USD` → `BTC`

#### Available Time Periods
- For hourly data: 1-90 days
- For daily data: 1-365 days

#### Available Intervals
- `hourly` or `1h` - Hourly data (limited to 90 days)
- `daily` or `1d` - Daily data (up to 365 days)

### Data Handling

The CoinGecko API provides the following data:
- Timestamp and close price
- Volume data

However, it doesn't provide open, high, and low prices directly. To maintain compatibility with the existing codebase:
- Open prices are derived from previous close prices
- High and low prices are estimated as +/- 1% of close prices

### Error Handling

The CoinGecko integration includes robust error handling:
- Empty dataset detection and handling
- Exception catching for network issues
- Automatic fallback to simulation when data is unavailable
- Proper symbol format validation and conversion
- Rate limiting awareness

## Web Application Integration

### Overview

The project now includes an interactive web application that provides a user-friendly interface for the trading bot. This allows for easier data visualization, model training, and price prediction without requiring programming knowledge.

### Features

1. **Interactive Dashboard**: Visualize cryptocurrency price data and predictions
2. **Model Training Interface**: Train and evaluate models with different parameters
3. **Real-time Predictions**: Generate and visualize future price predictions
4. **Data Source Selection**: Choose between CoinGecko, Yahoo Finance, or simulated data
5. **Configuration Options**: Customize data parameters, model settings, and visualization options

### Running the Application

```bash
python app.py
```

The application will be available at http://localhost:5000 by default.

### Application Flow

1. Select cryptocurrency (e.g., BTC, ETH)
2. Choose time interval (hourly, daily)
3. Select data source or use stored data
4. View historical data and technical indicators
5. Train a new model or use a saved model
6. Generate and visualize price predictions
7. Export results and predictions

### Automated Processing

The application includes automated processing features:
- Scheduled data updates
- Periodic model retraining
- Automated trading signal generation
- Performance reporting and alerts

## Model Evaluation and Performance Metrics

### Overview

The project now includes comprehensive model evaluation capabilities to assess prediction accuracy and performance. These metrics help users understand the reliability of the predictions and make informed trading decisions.

### Implemented Metrics

1. **Mean Absolute Error (MAE)**: Average absolute difference between predictions and actual values
2. **Root Mean Square Error (RMSE)**: Square root of the average squared differences
3. **Mean Absolute Percentage Error (MAPE)**: Percentage error relative to actual values
4. **Directional Accuracy**: Percentage of correct price movement direction predictions
5. **Profit/Loss Simulation**: Simulated trading results based on model predictions

### Visualization

The evaluation results are visualized through:
- Error distribution charts
- Prediction vs. actual price comparison
- Confusion matrix for directional accuracy
- Profit/loss over time charts
- Performance comparison between different models

### Usage

```python
# Evaluate model performance
metrics = evaluate_model_performance(
    model=model,
    X_test=X_test,
    y_test=y_test,
    scaler=preprocessor.scaler
)

# Display evaluation results
print_evaluation_metrics(metrics)

# Visualize evaluation results
plot_evaluation_results(
    actual_prices=actual_prices,
    predicted_prices=predicted_prices,
    symbol=symbol,
    interval=interval
)
```

The evaluation system provides insights into model strengths and weaknesses, helping users refine their trading strategies based on prediction reliability.
