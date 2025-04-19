import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.utils.logger import setup_logging

logger = setup_logging('siamese_model')

class SiameseNetwork(nn.Module):
    def __init__(self, input_shape):
        super(SiameseNetwork, self).__init__()
        self.sequence_length, self.features = input_shape
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(128)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        
        # Batch normalization
        lstm_out = self.bn(lstm_out)
        
        # Fully connected layers
        out = self.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        return out

    def forward(self, input1, input2):
        # Get embeddings for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Compute L1 distance
        distance = torch.abs(output1 - output2)
        
        # Final prediction
        prediction = self.sigmoid(self.fc2(distance))
        return prediction

class SiameseModel:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sequence_length = 8
        self.features = len(config['model']['siamese']['input_shape'][0])

    def set_seed(self, seed=33):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def build_model(self):
        """Build the Siamese network architecture."""
        try:
            self.set_seed()
            self.model = SiameseNetwork((self.sequence_length, self.features))
            self.model.to(self.device)
            
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['model']['siamese']['learning_rate']
            )
            
            logger.info("Siamese network model built successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise

    def train(self, left_input, right_input, targets, validation_split=0.15):
        """Train the Siamese network."""
        try:
            # Convert numpy arrays to PyTorch tensors
            left_input = torch.FloatTensor(left_input).to(self.device)
            right_input = torch.FloatTensor(right_input).to(self.device)
            targets = torch.FloatTensor(targets).to(self.device)
            
            # Create dataset and dataloader
            dataset = TensorDataset(left_input, right_input, targets)
            train_size = int((1 - validation_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['model']['siamese']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['model']['siamese']['batch_size'],
                shuffle=False
            )
            
            # Training loop
            for epoch in range(self.config['model']['siamese']['epochs']):
                self.model.train()
                train_loss = 0.0
                
                for batch_left, batch_right, batch_targets in train_loader:
                    self.optimizer.zero_grad()
                    
                    predictions = self.model(batch_left, batch_right)
                    loss = self.criterion(predictions, batch_targets.view(-1, 1))
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_left, batch_right, batch_targets in val_loader:
                        predictions = self.model(batch_left, batch_right)
                        loss = self.criterion(predictions, batch_targets.view(-1, 1))
                        val_loss += loss.item()
                
                logger.info(f"Epoch {epoch+1}/{self.config['model']['siamese']['epochs']} - "
                          f"Train Loss: {train_loss/len(train_loader):.4f} - "
                          f"Val Loss: {val_loss/len(val_loader):.4f}")
            
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, left_input, right_input):
        """Make predictions using the trained model."""
        try:
            self.model.eval()
            with torch.no_grad():
                left_input = torch.FloatTensor(left_input).to(self.device)
                right_input = torch.FloatTensor(right_input).to(self.device)
                predictions = self.model(left_input, right_input)
                return predictions.cpu().numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save(self, filepath):
        """Save the model to disk."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, filepath):
        """Load a saved model from disk."""
        try:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 