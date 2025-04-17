# predictor.py
class PricePredictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict_next_price(self, current_sequence):
        """Predict the next price given the current sequence"""
        # Ensure sequence is scaled
        prediction = self.model.predict(current_sequence, verbose=0)
        # Convert prediction back to original scale
        return self.preprocessor.inverse_transform(prediction)[0]
        
    def forecast_future_prices(self, df, steps=24):
        """
        Generate a forecast for future prices
        
        Args:
            df: DataFrame with historical data
            steps: Number of future steps to forecast
            
        Returns:
            DataFrame with forecasted prices
        """
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        
        # Get the last timestamp and prepare for forecasting
        last_timestamp = df['timestamp'].iloc[-1]
        latest_data = df.copy()
        
        # Prepare data for the first prediction
        X, _ = self.preprocessor.prepare_data(latest_data)
        last_sequence = X[-1:]
        
        # Initialize forecast dataframe
        forecast_timestamps = []
        forecast_prices = []
        
        # Determine time interval from the data
        if len(df) >= 2:
            avg_interval = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[-2]).total_seconds() / 3600
            if avg_interval >= 20:  # Daily
                time_delta = timedelta(days=1)
            elif avg_interval >= 3:  # 4h
                time_delta = timedelta(hours=4)
            else:  # 1h
                time_delta = timedelta(hours=1)
        else:
            # Default to 1h if we can't determine
            time_delta = timedelta(hours=1)
        
        # Generate forecasts
        for i in range(steps):
            # Predict next price
            next_price = self.predict_next_price(last_sequence)
            
            # Calculate next timestamp
            next_timestamp = last_timestamp + time_delta
            
            # Store prediction
            forecast_timestamps.append(next_timestamp)
            forecast_prices.append(next_price)
            
            # Update for next iteration
            last_timestamp = next_timestamp
            
            # Create new row to append
            new_row = pd.DataFrame({
                'timestamp': [next_timestamp],
                'open': [next_price],
                'high': [next_price * 1.005],  # Slight increase for simulation
                'low': [next_price * 0.995],   # Slight decrease for simulation
                'close': [next_price],
                'volume': [latest_data['volume'].tail(30).mean()]  # Use recent average volume
            })
            
            # Update dataframe with new prediction
            latest_data = pd.concat([latest_data, new_row], ignore_index=True)
            
            # Recalculate features and get new sequence
            X, _ = self.preprocessor.prepare_data(latest_data)
            last_sequence = X[-1:]
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': forecast_timestamps,
            'price': forecast_prices
        })
        
        return forecast_df

