"""
Visualization Module

This module provides functions for plotting and visualizing cryptocurrency data,
including price charts, volume bars, and model predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
import matplotlib.dates as mdates
from matplotlib.figure import Figure


def setup_price_figure(figsize: Tuple[int, int] = (12, 8)) -> Tuple[Figure, Any, Any]:
    """
    Setup a figure for plotting price and volume data.
    
    Args:
        figsize: Figure size as (width, height)
        
    Returns:
        Tuple of (figure, price_axis, volume_axis)
    """
    # Create figure and subplots with appropriate size ratio between price and volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                  gridspec_kw={'height_ratios': [3, 1]})
    
    # Configure price subplot
    ax1.grid(True, alpha=0.3)
    
    # Configure volume subplot
    ax2.grid(True, alpha=0.3)
    
    return fig, ax1, ax2


def format_axes(fig: Figure, price_ax: Any, vol_ax: Any, symbol: str, interval: str) -> None:
    """
    Format axes with proper labels, titles and styling.
    
    Args:
        fig: Figure to format
        price_ax: Price axis to format
        vol_ax: Volume axis to format
        symbol: Symbol being plotted
        interval: Interval of the data
    """
    # Set titles and labels
    price_ax.set_title(f'{symbol} Price Chart ({interval})')
    price_ax.set_ylabel('Price (USD)')
    vol_ax.set_ylabel('Volume')
    
    # Format axes
    # Rotate x-ticks for better readability
    plt.setp(price_ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(vol_ax.get_xticklabels(), rotation=45, ha='right')
    
    # Format datetime x-axis
    date_format = mdates.DateFormatter('%Y-%m-%d')
    price_ax.xaxis.set_major_formatter(date_format)
    vol_ax.xaxis.set_major_formatter(date_format)
    
    # Add grid to both axes
    price_ax.grid(True, alpha=0.3)
    vol_ax.grid(True, alpha=0.3)
    
    # Adjust layout to prevent overlapping
    fig.tight_layout()


def plot_price_data(ax: Any, timestamps: Union[List[Any], pd.Series], prices: Union[List[float], pd.Series], 
                   label: str = 'Historical Price', color: str = 'blue') -> None:
    """
    Plot price data on the given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        timestamps: List of timestamps for x-axis
        prices: List of prices for y-axis
        label: Label for the line
        color: Color for the line
    """
    ax.plot(timestamps, prices, label=label, color=color)


def plot_volume_data(ax: Any, timestamps: Union[List[Any], pd.Series], volumes: Union[List[float], pd.Series], 
                    color: str = 'green', alpha: float = 0.5) -> None:
    """
    Plot volume data as bars on the given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        timestamps: List of timestamps for x-axis
        volumes: List of volumes for y-axis
        color: Color for the bars
        alpha: Transparency level
    """
    ax.bar(timestamps, volumes, color=color, alpha=alpha)


def plot_predictions(ax: Any, timestamps: Union[List[Any], pd.Series, pd.DatetimeIndex], prices: Union[List[float], pd.Series], 
                    label: str = 'Model Prediction', color: str = 'red', 
                    linestyle: str = '--') -> None:
    """
    Plot model predictions on the given axis.
    
    Args:
        ax: Matplotlib axis to plot on
        timestamps: List of timestamps for x-axis
        prices: List of prices for y-axis
        label: Label for the line
        color: Color for the line
        linestyle: Line style (solid, dashed, etc.)
    """
    ax.plot(timestamps, prices, label=label, color=color, linestyle=linestyle)


def add_vertical_marker(ax: Any, timestamp: Any, color: str = 'gray', 
                       linestyle: str = '--', alpha: float = 0.7) -> None:
    """
    Add a vertical line marker at a specific timestamp.
    
    Args:
        ax: Matplotlib axis to plot on
        timestamp: Timestamp position for the vertical line
        color: Color for the line
        linestyle: Line style
        alpha: Transparency level
    """
    ax.axvline(x=timestamp, color=color, linestyle=linestyle, alpha=alpha)


def debug_price_ranges(historical_prices: List[float], predicted_prices: List[float]) -> None:
    """
    Print debug information about price ranges to help diagnose visualization issues.
    
    Args:
        historical_prices: List of historical prices
        predicted_prices: List of predicted prices
    """
    hist_min, hist_max = min(historical_prices), max(historical_prices)
    print(f"Historical price range: min=${hist_min:.2f}, max=${hist_max:.2f}")
    
    if predicted_prices:
        pred_min, pred_max = min(predicted_prices), max(predicted_prices)
        print(f"Predicted price range: min=${pred_min:.2f}, max=${pred_max:.2f}")
        
        # Check if predicted range is substantially different
        if abs(pred_max - pred_min) < 0.01 * abs(hist_max - hist_min):
            print("WARNING: Predicted prices have very small variation compared to historical data!")
            print("This could make them appear as a constant line on the graph.")


def plot_crypto_data_with_predictions(df: pd.DataFrame, symbol: str, interval: str,
                                     pred_dates: Optional[pd.Series] = None, 
                                     predicted_prices: Optional[List[float]] = None,
                                     future_dates: Optional[pd.DatetimeIndex] = None,
                                     future_predictions: Optional[List[float]] = None) -> None:
    """
    Create a comprehensive plot of cryptocurrency data with predictions.
    
    Args:
        df: DataFrame with historical price data
        symbol: Symbol being plotted (e.g., 'BTCUSDT')
        interval: Time interval (e.g., '1h', '4h', '1d')
        pred_dates: Optional dates for model predictions on test data
        predicted_prices: Optional prices predicted by model on test data
        future_dates: Optional dates for future predictions
        future_predictions: Optional future price predictions
    """
    # Setup the figure
    fig, price_ax, vol_ax = setup_price_figure()
    
    # Plot historical price data
    plot_price_data(price_ax, df['timestamp'], df['close'])
    
    # Plot volume data
    plot_volume_data(vol_ax, df['timestamp'], df['volume'])
    
    # Debug price ranges
    debug_price_ranges(df['close'].tolist(), 
                      (predicted_prices if predicted_prices is not None else []) + 
                      (future_predictions if future_predictions is not None else []))
    
    # Plot test predictions if available
    if pred_dates is not None and predicted_prices is not None and len(pred_dates) == len(predicted_prices):
        plot_predictions(price_ax, pred_dates, predicted_prices)
        print(f"Plotted {len(predicted_prices)} predicted prices")
        
        # Add vertical marker for prediction start
        if len(pred_dates) > 0:
            add_vertical_marker(price_ax, pred_dates.iloc[0])
    
    # Plot future predictions if available
    if future_dates is not None and future_predictions is not None and len(future_dates) == len(future_predictions):
        plot_predictions(price_ax, future_dates, future_predictions, 
                        label='Future Prediction', linestyle='-')
        print(f"Plotted {len(future_predictions)} future predictions")
        
        # Add vertical marker for future prediction start
        if len(future_dates) > 0:
            add_vertical_marker(price_ax, future_dates[0])
    
    # Add legend
    price_ax.legend()
    
    # Format axes
    format_axes(fig, price_ax, vol_ax, symbol, interval)
    
    # Show the plot
    plt.show()


def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot training history metrics.
    
    Args:
        history: History dictionary from model training
    """
    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    epochs = range(1, len(history['loss']) + 1)
    ax1.plot(epochs, history['loss'], 'b', label='Training loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot learning rate if available
    if 'lr' in history:
        ax2.plot(epochs, history['lr'], 'g', label='Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show() 