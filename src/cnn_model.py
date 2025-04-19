import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

class CNNModel(nn.Module):
    def __init__(self, input_shape=None, num_classes=None, config=None):
        super(CNNModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        
        # Initialize with default values if not provided
        if input_shape is not None and num_classes is not None and config is not None:
            self._initialize_model()
            
    def _initialize_model(self):
        """Initialize model architecture"""
        # Get input dimensions from config
        self.sequence_length = self.input_shape[0]  # sequence length
        self.in_channels = 18  # Fixed to match data features
        
        # Calculate padding to maintain sequence length
        kernel_size = 3
        padding = kernel_size // 2
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(in_channels=18, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_flattened_size()
        logger.info(f"Flattened feature size: {self.feature_size}")
        
        # Fully connected layers
        self.dense_layers = nn.Sequential(
            nn.Linear(128, 256),  # Fixed input size to match conv output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _get_flattened_size(self):
        # Create a dummy input to calculate the size after convolutions
        x = torch.randn(1, 18, self.sequence_length)
        logger.info(f"Initial input shape: {x.shape}")
        
        # Log shape after each layer
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            logger.info(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        # Calculate the size after the last conv layer
        last_conv_shape = x.shape
        logger.info(f"Last conv layer shape: {last_conv_shape}")
        
        # The flattened size should be the number of channels in the last conv layer
        flattened_size = last_conv_shape[1]  # Number of channels
        logger.info(f"Using flattened size: {flattened_size}")
        return flattened_size
        
    def forward(self, x):
        # Ensure input is the right shape (batch_size, channels, sequence_length)
        if len(x.shape) == 2:
            # Input is [batch_size, features] -> [batch_size, features, 1]
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            # Input is [batch_size, sequence_length, features] -> [batch_size, features, sequence_length]
            x = x.permute(0, 2, 1)
            
        # Log input shape
        logger.info(f"Input shape: {x.shape}")
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        logger.info(f"After conv layers shape: {x.shape}")
        
        # Global average pooling to reduce to [batch_size, channels]
        x = torch.mean(x, dim=2)
        logger.info(f"After global average pooling shape: {x.shape}")
        
        # Apply dense layers
        x = self.dense_layers(x)
        logger.info(f"After dense layers shape: {x.shape}")
        
        return x
        
    def compile_model(self, learning_rate=0.001):
        """Set up optimizer and loss function"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def train_model(self, train_loader, val_loader=None, epochs=50):
        """Train the model using a DataLoader with early stopping"""
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Early stopping parameters
        patience = self.config['model']['cnn']['early_stopping']['patience']
        min_delta = self.config['model']['cnn']['early_stopping']['min_delta']
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_labels in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_labels.long())
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_labels in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                        
                        outputs = self(batch_X)
                        loss = self.criterion(outputs, batch_labels.long())
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                # Calculate validation metrics
                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                
                # Early stopping check
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = self.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f'Early stopping triggered at epoch {epoch+1}')
                        if best_weights is not None:
                            self.load_state_dict(best_weights)
                        break
            else:
                val_loss = None
                val_acc = None
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            if val_loss is not None:
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # Log progress
            logger.info(f'Epoch {epoch+1}/{epochs}')
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            if val_loss is not None:
                logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return history
    
    def plot_training_history(self, history, save_path):
        """Plot and save training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('CNN Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_confusion_matrix(self, true_labels, predictions, save_path):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('CNN Model Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_path)
        plt.close()

    def plot_classification_results(self, true_labels, predictions, save_path):
        """Plot and save classification results"""
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(true_labels)), true_labels, label='True Labels', alpha=0.5)
        plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.5)
        plt.title('CNN Model Classification Results')
        plt.xlabel('Sample Index')
        plt.ylabel('Class')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def evaluate(self, test_loader):
        """Evaluate the model on test data and generate visualizations"""
        self.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_labels in test_loader:
                batch_X = batch_X.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Generate and save visualizations
        results_dir = Path(self.config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            all_labels,
            all_predictions,
            results_dir / 'cnn_confusion_matrix.png'
        )
        
        # Plot classification results
        self.plot_classification_results(
            all_labels,
            all_predictions,
            results_dir / 'cnn_classification_results.png'
        )
        
        # Print classification report
        report = classification_report(all_labels, all_predictions)
        logger.info("\nCNN Model Classification Report:")
        logger.info(report)
        
        # Save classification report
        with open(results_dir / 'cnn_classification_report.txt', 'w') as f:
            f.write(report)
        
        return accuracy, np.array(all_predictions), np.array(all_labels)
    
    def save_model(self, filepath):
        """Save the model to a file"""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'config': self.config
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """Load model from file"""
        try:
            logger.info(f"Loading model from {filepath}")
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
            
            # Update model parameters
            self.input_shape = checkpoint['input_shape']
            self.num_classes = checkpoint['num_classes']
            self.config = checkpoint['config']
            
            logger.info(f"Model parameters: input_shape={self.input_shape}, num_classes={self.num_classes}")
            
            # Initialize model architecture
            self._initialize_model()
            
            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if available
            if checkpoint['optimizer_state_dict'] is not None and self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.eval()
            logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict(self, x):
        """
        Make predictions using the trained model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, features]
            
        Returns:
            torch.Tensor: Class probabilities
        """
        self.eval()
        with torch.no_grad():
            # Convert input to tensor if it's a numpy array
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
                
            # Move tensor to the same device as the model
            x = x.to(self.device)
                
            # Ensure correct shape [batch_size, sequence_length, features]
            if len(x.shape) == 2:
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Get predictions
            outputs = self(x)
            probabilities = F.softmax(outputs, dim=1)
            
        return probabilities 

    def plot_roc_curve(self, test_loader, save_path):
        """Plot and save ROC curve"""
        self.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_labels in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self(batch_X)
                probs = torch.softmax(outputs, dim=1)
                
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate ROC curve for each class
        plt.figure(figsize=(10, 6))
        for i in range(self.num_classes):
            fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"ROC curve saved to {save_path}") 