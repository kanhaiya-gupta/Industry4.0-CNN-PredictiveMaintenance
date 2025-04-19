import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from .data_processor import DataProcessor
from .siamese_model import SiameseNetwork
from .evaluation import ModelEvaluator
from .visualization import ModelVisualizer
from .cnn_model import CNNModel
import numpy as np
from src.config_loader import ConfigLoader
from pathlib import Path
from src.utils.logger import setup_logging

# Initialize logger
logger = setup_logging('training')

def train_and_evaluate():
    # Load configuration
    config = ConfigLoader()
    
    # Get paths from config
    paths = config.get_path_config()
    data_path = Path(paths['data_dir']) / paths['sensor_data_file']
    
    # Initialize data processor
    data_processor = DataProcessor(data_path)
    
    # Load and preprocess data
    X, y = data_processor.load_data()
    X_train, X_test, y_train, y_test = data_processor.preprocess_data((X, y))
    
    # Get model configurations
    cnn_config = config.get_model_config('cnn')
    siamese_config = config.get_model_config('siamese')
    
    # Get the number of classes from the data
    num_classes = len(np.unique(y_train))
    logger.info(f"Number of classes: {num_classes}")
    
    # Train CNN model
    input_channels = X_train.shape[1]  # number of features
    sequence_length = X_train.shape[2]  # sequence length (1 in this case)
    cnn_model = CNNModel(input_shape=(input_channels, sequence_length), num_classes=num_classes)
    cnn_model.compile_model()
    history_cnn = cnn_model.train_model(X_train, y_train, X_test, y_test, epochs=cnn_config['epochs'])
    
    # Train Siamese network
    logger.info("\nTraining Siamese Network...")
    siamese_model = SiameseNetwork(input_shape=(X_train.shape[1], 1))  # (num_features, sequence_length)
    siamese_model.compile_model(learning_rate=siamese_config['learning_rate'])
    
    # Create training pairs and labels
    logger.info("Creating Siamese pairs...")
    pairs, labels = data_processor.create_siamese_pairs(X_train, y_train)
    
    # Convert pairs to the correct format for the model
    pairs_tensor = torch.FloatTensor(pairs).reshape(-1, 2, X_train.shape[1], 1)  # Shape: [num_pairs, 2, num_features, 1]
    labels_tensor = torch.FloatTensor(labels)
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(pairs_tensor, labels_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=siamese_config['batch_size'], 
        shuffle=True
    )
    
    # Train the model
    logger.info("Training Siamese model...")
    history_siamese = siamese_model.train_model(train_loader, epochs=siamese_config['epochs'])
    
    # Initialize evaluators and visualizers
    cnn_evaluator = ModelEvaluator(cnn_model, data_processor)
    siamese_evaluator = ModelEvaluator(siamese_model, data_processor)
    cnn_visualizer = ModelVisualizer(cnn_model)
    siamese_visualizer = ModelVisualizer(siamese_model)
    
    # Evaluate CNN model
    logger.info("\nEvaluating CNN Model...")
    cnn_metrics = cnn_evaluator.evaluate_model(X_test, y_test, model_type='cnn')
    cnn_visualizer.plot_evaluation_metrics(cnn_metrics, "CNN Model")
    cnn_visualizer.plot_training_history(history_cnn)
    
    # Evaluate Siamese model
    logger.info("\nEvaluating Siamese Model...")
    siamese_metrics = siamese_evaluator.evaluate_model(X_test, y_test, model_type='siamese')
    siamese_visualizer.plot_evaluation_metrics(siamese_metrics, "Siamese Model")
    siamese_visualizer.plot_training_history(history_siamese)
    
    # Compare models
    cnn_visualizer.compare_models([cnn_model, siamese_model], X_test, y_test, ["CNN", "Siamese"])
    
    # Save models
    models_dir = Path(paths['models_dir'])
    models_dir.mkdir(exist_ok=True)
    
    # Save CNN model
    cnn_model_path = models_dir / paths['cnn_model_file']
    cnn_model.save_model(cnn_model_path)
    logger.info(f"CNN model saved to {cnn_model_path}")
    
    # Save Siamese model
    siamese_model_path = models_dir / paths['siamese_model_file']
    siamese_model.save_model(siamese_model_path)
    logger.info(f"Siamese model saved to {siamese_model_path}")
    
    # Verify models were saved
    if cnn_model_path.exists():
        logger.info("CNN model file verified")
    else:
        logger.error("CNN model file not found after saving")
        
    if siamese_model_path.exists():
        logger.info("Siamese model file verified")
    else:
        logger.error("Siamese model file not found after saving")

if __name__ == "__main__":
    train_and_evaluate()
