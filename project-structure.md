# Cryptocurrency Trading Bot Project Structure

## Files and Components

### Core Components

- **app.py**: Main web application and user interface
- **main.py**: Command-line interface and entry point for batch processing
- **data_collector.py**: Collects data from Binance API
- **data_downloader.py**: Downloads data from CoinGecko and Yahoo Finance
- **data_simulator.py**: Generates synthetic market data for testing
- **data_manager.py**: Manages data persistence and organization
- **data_storage.py**: Handles low-level storage operations
- **data_utils.py**: Utility functions for data processing
- **data_preprocessor.py**: Prepares data for model training
- **feature_engineer.py**: Calculates technical indicators
- **model_builder.py**: Defines model architecture
- **model_utils.py**: Functions for model training and evaluation
- **predictor.py**: Makes price predictions using trained models
- **visualization.py**: Creates charts and visualizations

### Configuration and Environment

- **requirements.txt**: Python dependencies
- **.env**: Environment variables and API keys
- **pyrightconfig.json**: Type checking configuration

### Data Storage

- **crypto_data/**: Root data directory
  - **raw/**: Raw market data
  - **processed/**: Processed datasets
  - **cache/**: Temporary data
  - **metadata/**: Data metadata files

### Model Storage

- **saved_models/**: Directory for trained models
  - **BTCUSDT_1h/**: Models for BTC hourly data
  - **ETHUSDT_1d/**: Models for ETH daily data
  - etc.

## Project Flow

### 1. Data Collection and Management

```
┌─────────────────┐     ┌─────────────────┐      ┌─────────────────┐
│   CoinGecko API │     │  Yahoo Finance  │      │   Binance API   │
└────────┬────────┘     └────────┬────────┘      └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────┐     ┌─────────────────┐      ┌─────────────────┐
│download_from_   │     │download_from_   │      │BinanceCollector │
│coingecko()      │     │yahoo()          │      │                 │
└────────┬────────┘     └────────┬────────┘      └────────┬────────┘
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐      ┌─────────────────┐
                        │  DataCollector  │◄─────┤ DataSimulator   │
                        │                 │      │                 │
                        └────────┬────────┘      └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  DataManager    │
                        │                 │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Data Storage   │
                        │  (crypto_data/) │
                        └─────────────────┘
```

### 2. Data Processing and Model Training

```
┌─────────────────┐
│  Data Storage   │
│  (crypto_data/) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│FeatureEngineer  │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│DataPreprocessor │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  ModelBuilder   │────►│  saved_models/  │
│                 │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       │
┌─────────────────┐              │
│PricePredictor   │◄─────────────┘
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Predictions &   │
│ Trading Signals │
└─────────────────┘
```

### 3. User Interface and Visualization

```
┌─────────────────┐     ┌─────────────────┐
│     app.py      │     │     main.py     │
│  Web Interface  │     │  CLI Interface  │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────────────────┼───────────────────┐
                                 │                   │
                                 ▼                   ▼
                        ┌─────────────────┐ ┌─────────────────┐
                        │ model_utils.py  │ │ data_utils.py   │
                        │                 │ │                 │
                        └────────┬────────┘ └────────┬────────┘
                                 │                   │
                                 └───────────────────┼────────────────────┐
                                                     │                    │
                                                     ▼                    ▼
                                            ┌─────────────────┐ ┌─────────────────┐
                                            │visualization.py │ │ Data Processing │
                                            │                 │ │    Pipeline     │
                                            └────────┬────────┘ └─────────────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │ Interactive     │
                                            │ Visualizations  │
                                            └─────────────────┘
```

## Data Flow

1. **Data Collection**: 
   - Fetch data from CoinGecko (primary)
   - Fallback to Yahoo Finance if CoinGecko fails
   - Fallback to Binance if configured
   - Generate simulated data if all sources fail

2. **Data Storage**:
   - Save raw data to disk
   - Track metadata for each dataset
   - Implement caching for efficiency

3. **Feature Engineering**:
   - Calculate technical indicators
   - Create derived features
   - Normalize and scale data

4. **Model Training**:
   - Split data into training/validation/test sets
   - Train LSTM model
   - Validate model performance
   - Save model to disk

5. **Prediction**:
   - Load model from disk
   - Make price predictions
   - Generate trading signals
   - Evaluate prediction accuracy

6. **Visualization**:
   - Display historical prices
   - Show predicted prices
   - Visualize technical indicators
   - Chart trading performance
   - Create interactive dashboards

## Application Configuration Options

### Data Source Selection

```python
# In app.py or main.py
options = {
    "use_stored_only": True,  # Use only stored data
    "download_new": False,    # Don't download new data
    "use_best_data": True,    # Automatically select best dataset
    "use_saved_model": True,  # Use saved models if available
    "reset_models": False     # Don't reset existing models
}
```

### Model Configuration

```python
# Model hyperparameters
model_config = {
    "lstm_units": [50, 50, 50],
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "sequence_length": 60
}
```

### Trading Strategy Settings

```python
# Trading strategy parameters
strategy_config = {
    "threshold_buy": 0.02,    # 2% predicted increase
    "threshold_sell": 0.01,   # 1% predicted decrease
    "stop_loss": 0.05,        # 5% stop loss
    "take_profit": 0.1,       # 10% take profit
    "position_size": 0.1      # 10% of capital per trade
}
```

## Performance Optimization

- **Caching**: Frequently used data is cached to minimize redundant processing
- **Lazy Loading**: Data is loaded only when needed to reduce memory usage
- **Incremental Updates**: Only new data is downloaded to minimize API usage
- **Model Checkpointing**: Save models at checkpoints to resume training
- **Data Validation**: Check data integrity before using in critical operations

## Extension Points

The project is designed for extensibility:

1. **New Data Sources**: Add new source in data_downloader.py
2. **New Models**: Implement in model_builder.py
3. **New Features**: Add to feature_engineer.py
4. **New Visualizations**: Create in visualization.py
5. **Trading Strategies**: Implement in a new module 