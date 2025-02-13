# predictor.py
class PricePredictor:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def predict_next_price(self, current_sequence):
        """Predict the next price given the current sequence"""
        # Ensure sequence is scaled
        prediction = self.model.predict(current_sequence)
        # Convert prediction back to original scale
        return self.preprocessor.inverse_transform(prediction)[0]

