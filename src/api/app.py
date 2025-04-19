from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.config_loader import ConfigLoader
from src.train import train_and_evaluate
from src.data_processor import DataProcessor
from src.visualization import ModelVisualizer
from src.evaluation import ModelEvaluator
import torch
from src.siamese_model import SiameseNetwork
from src.cnn_model import CNNModel
from contextlib import asynccontextmanager
import os
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Union
import yaml
from src.utils.logger import setup_logging

# Initialize logger
logger = setup_logging('api')

# Load configuration
config_path = Path('config/config.yaml')
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Define input schema with constraints
class SensorInput(BaseModel):
    I_w_BLO_Weg: float = Field(..., description="BLO Weg current")
    O_w_BLO_power: float = Field(..., description="BLO power output")
    O_w_BLO_voltage: float = Field(..., description="BLO voltage output")
    I_w_BHL_Weg: float = Field(..., description="BHL Weg current")
    O_w_BHL_power: float = Field(..., description="BHL power output")
    O_w_BHL_voltage: float = Field(..., description="BHL voltage output")
    I_w_BHR_Weg: float = Field(..., description="BHR Weg current")
    O_w_BHR_power: float = Field(..., description="BHR power output")
    O_w_BHR_voltage: float = Field(..., description="BHR voltage output")
    I_w_BRU_Weg: float = Field(..., description="BRU Weg current")
    O_w_BRU_power: float = Field(..., description="BRU power output")
    O_w_BRU_voltage: float = Field(..., description="BRU voltage output")
    I_w_HR_Weg: float = Field(..., description="HR Weg current")
    O_w_HR_power: float = Field(..., description="HR power output")
    O_w_HR_voltage: float = Field(..., description="HR voltage output")
    I_w_HL_Weg: float = Field(..., description="HL Weg current")
    O_w_HL_power: float = Field(..., description="HL power output")
    O_w_HL_voltage: float = Field(..., description="HL voltage output")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "I_w_BLO_Weg": -107,
            "O_w_BLO_power": 0,
            "O_w_BLO_voltage": 0,
            "I_w_BHL_Weg": 0,
            "O_w_BHL_power": 0,
            "O_w_BHL_voltage": 0,
            "I_w_BHR_Weg": -1268,
            "O_w_BHR_power": 0,
            "O_w_BHR_voltage": 0,
            "I_w_BRU_Weg": -26,
            "O_w_BRU_power": 84,
            "O_w_BRU_voltage": 11,
            "I_w_HR_Weg": 0,
            "O_w_HR_power": 7168,
            "O_w_HR_voltage": 26,
            "I_w_HL_Weg": 0,
            "O_w_HL_power": 7720,
            "O_w_HL_voltage": 24
        }
    })

class PredictionResponse(BaseModel):
    cnn_prediction: List[float] = Field(..., description="CNN model prediction probabilities")
    siamese_prediction: List[float] = Field(..., description="Siamese model prediction probabilities")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "cnn_prediction": [0.1, 0.8, 0.1],
            "siamese_prediction": [0.2, 0.7, 0.1]
        }
    })

# Global variables for models
cnn_model = None
siamese_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    load_models()
    yield
    # Shutdown
    # Clean up resources if needed

def load_models():
    """Load trained models"""
    global cnn_model, siamese_model
    try:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Checking models in directory: {models_dir}")
        
        # Check if model files exist
        cnn_model_path = models_dir / 'cnn_model.pth'
        siamese_model_path = models_dir / 'siamese_model.pth'
        
        logger.info(f"Looking for CNN model at: {cnn_model_path}")
        logger.info(f"Looking for Siamese model at: {siamese_model_path}")
        
        if os.path.exists(cnn_model_path):
            logger.info("Found CNN model file, loading...")
            # Load the checkpoint
            checkpoint = torch.load(cnn_model_path, weights_only=True)
            # Extract the model state dict and config
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    # Get the correct input shape and num_classes from the saved model
                    input_shape = checkpoint.get('input_shape', config['model']['cnn']['input_shape'])
                    num_classes = checkpoint.get('num_classes', config['model']['cnn']['num_classes'])
                else:
                    state_dict = checkpoint
                    # If no config in checkpoint, use the saved model's parameters
                    input_shape = (19, 1)  # Based on the saved model's conv layer
                    num_classes = 276  # Based on the saved model's output layer
            else:
                state_dict = checkpoint
                input_shape = (19, 1)
                num_classes = 276
                
            logger.info(f"Creating CNN model with input_shape={input_shape}, num_classes={num_classes}")
            # Create a new model instance with the correct architecture
            cnn_model = CNNModel(
                input_shape=input_shape,
                num_classes=num_classes
            )
            # Load the state dict into the model
            cnn_model.load_state_dict(state_dict)
            cnn_model.eval()
            logger.info("CNN model loaded successfully")
        else:
            logger.warning("CNN model not found. It will be created during training.")
            
        if os.path.exists(siamese_model_path):
            logger.info("Found Siamese model file, loading...")
            try:
                # Create a temporary model to get the input shape
                temp_model = SiameseNetwork(input_shape=(19, 1))
                siamese_model = temp_model
                siamese_model.load_model(siamese_model_path)
                siamese_model.eval()
                logger.info("Siamese model loaded successfully")
            except Exception as e:
                logger.error(f"Could not load Siamese model: {str(e)}")
                logger.warning("A new Siamese model will be created during training.")
                siamese_model = None
        else:
            logger.warning("Siamese model not found. It will be created during training.")
            
        # Validate loaded models
        if cnn_model is not None and siamese_model is not None:
            logger.info("\nModel validation:")
            logger.info(f"CNN Model type: {type(cnn_model)}")
            logger.info(f"Siamese Model type: {type(siamese_model)}")
            logger.info("Both models loaded and validated successfully")
        else:
            logger.warning("\nWarning: Not all models are loaded")
            
    except Exception as e:
        error_msg = f"Error loading models: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Industry 4.0 Predictive Maintenance API",
    description="API for predictive maintenance using CNN and Siamese Networks",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Industry 4.0 Predictive Maintenance API"}

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check API health status"""
    try:
        status = {
            "status": "healthy",
            "cnn_model_loaded": bool(cnn_model),
            "siamese_model_loaded": bool(siamese_model),
            "config_loaded": bool(config)
        }
        logger.info(f"Health check completed: {status}")
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Health check error: {str(e)}"
        )

@app.post("/train")
async def train_models():
    """Train both CNN and Siamese models"""
    try:
        logger.info("Starting model training...")
        train_and_evaluate()
        logger.info("Training completed, loading models...")
        load_models()  # Reload models after training
        return {"message": "Training completed successfully"}
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: SensorInput) -> Dict[str, List[float]]:
    """Make predictions using the trained models"""
    try:
        if cnn_model is None or siamese_model is None:
            error_msg = "Models not loaded. Please train the models first using the /train endpoint."
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)
            
        logger.info("Making predictions...")
        # Get data path from config
        data_path = Path('data/sensor_data.csv')
        
        # Process input data
        data_processor = DataProcessor(data_path)
        
        # Convert input data to numpy array
        input_data = data.model_dump()
        features = []
        for key in sorted(input_data.keys()):
            features.append(float(input_data[key]))
        
        # Convert to numpy array and reshape for the model
        X = np.array(features).reshape(1, -1)  # Shape: (1, num_features)
        
        # Preprocess the data
        processed_data = data_processor.preprocess_data(X)
        
        # Convert to tensor
        processed_data = torch.FloatTensor(processed_data)
        
        # Make predictions
        with torch.no_grad():
            cnn_prediction = cnn_model(processed_data)
            siamese_prediction = siamese_model(processed_data)
            
        logger.info("Predictions completed successfully")
        return {
            "cnn_prediction": cnn_prediction.tolist(),
            "siamese_prediction": siamese_prediction.tolist()
        }
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg) 