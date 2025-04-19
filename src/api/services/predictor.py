import numpy as np
import pandas as pd
from src.data_processor import DataProcessor
from src.cnn_model import CNNModel
from src.config_loader import ConfigLoader
import os

class PredictorService:
    def __init__(self):
        self.config = ConfigLoader()
        self.data_processor = None
        self.model = None
        self._initialize_components()

    def _initialize_components(self):
        # Get paths from config
        paths = self.config.get_path_config()
        data_path = os.path.join(paths['data_dir'], paths['sensor_data_file'])
        model_path = os.path.join(paths['models_dir'], paths['cnn_model_file'])
        
        # Initialize data processor
        self.data_processor = DataProcessor(data_path)
        
        # Load and preprocess data
        data = self.data_processor.load_data()
        X_train, _ = self.data_processor.preprocess_data(data)
        
        # Initialize model with config
        model_config = self.config.get_model_config('cnn')
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = model_config['num_classes']
        self.model = CNNModel(input_shape, num_classes)
        
        # Load the trained model
        if os.path.exists(model_path):
            self.model.load_model(model_path)
        else:
            raise Exception("Model file not found")

    def predict(self, file_content: bytes) -> dict:
        try:
            # Read and process the uploaded file
            df = pd.read_csv(pd.io.common.BytesIO(file_content))
            
            # Preprocess the data
            X = self.data_processor.scaler.transform(df)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Make predictions
            predictions = self.model.predict(X)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Get threshold from config
            threshold = self.config.get_evaluation_config()['decision_threshold']
            
            # Prepare response
            results = []
            for i, pred in enumerate(predicted_classes):
                confidence = float(predictions[i][pred])
                if confidence >= threshold:
                    results.append({
                        "sample_id": i,
                        "predicted_class": int(pred),
                        "confidence": confidence
                    })
            
            return {"predictions": results}
        
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}") 