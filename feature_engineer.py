# feature_engineer.py
class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df):
        """Add technical indicators to the dataset"""
        # Moving averages
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA14'] = df['close'].rolling(window=14).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['close'].rolling(window=14).std()
        
        return df