"""
Cryptocurrency Price Prediction Tool - Main Module

This module provides the main entry points for the cryptocurrency price prediction 
application. It serves as a backward-compatible wrapper around the new modular 
implementation. For new code, consider using app.py directly.
"""

import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crypto_prediction")

# Import functions from our new modular structure
try:
    from app import run_with_options, main
    from data_downloader import (
        download_from_coingecko, 
        download_and_store_historical_data
    )
    from data_utils import (
        view_stored_data, 
        list_available_data, 
        find_cleanest_dataset
    )
    from model_utils import (
        save_model, 
        load_saved_model, 
        delete_saved_models
    )
    
    logger.info("Using modular application structure")
    
except ImportError as e:
    logger.error(f"Error importing modular components: {str(e)}")
    logger.error("Falling back to legacy implementation")
    
    # If imports fail, we're running the original main.py
    # Leave existing code in place
    pass


if __name__ == "__main__":
    # Parse command line arguments if provided
    # This is a simple example - you could expand with argparse for more options
    use_stored_only = False
    download_new = True
    list_only = False
    use_combined_timeframes = True
    use_best_data = False
    reset_models = False
    use_saved_model = True
    
    if len(sys.argv) > 1:
        if "--list-only" in sys.argv:
            list_only = True
        if "--download-new" in sys.argv:
            download_new = True
        if "--use-stored-only" in sys.argv:
            use_stored_only = True
        if "--reset-models" in sys.argv:
            reset_models = True
        if "--use-pretrained-only" in sys.argv:  # New option
            use_pretrained_only = True
            use_saved_model = True  # Force this to be true
        if "--pretrain-all" in sys.argv:  # New option
            pretrain_all = True
        if "--use-combined-timeframes" in sys.argv:  # New option
            use_combined_timeframes = True
        if "--help" in sys.argv:
            print("Cryptocurrency Price Prediction Tool")
            print("Usage: python main.py [options]")
            print("Options:")
            print("  --list-only               Only list available data without analysis")
            print("  --download-new            Download fresh data")
            print("  --use-stored-only         Only use stored data without downloading")
            print("  --reset-models            Delete existing models and train new ones")
            print("  --use-pretrained-only     Only use pre-trained models, don't train new ones")
            print("  --pretrain-all            Pre-train models for all intervals before analysis")
            print("  --use-combined-timeframes Join 1h, 4h, and 1d data before training")
            print("  --help                    Display this help message")
            sys.exit(0)
    
    try:
        # Try to use the new modular implementation
        run_with_options(
            use_stored_only=use_stored_only,
            download_new=download_new,
            list_only=list_only,
            use_best_data=use_best_data,
            use_saved_model=use_saved_model,
            reset_models=reset_models,
            use_combined_timeframes=use_combined_timeframes
        )
    except NameError:
        # If run_with_options is not defined, we're in legacy mode
        logger.warning("Using legacy implementation")