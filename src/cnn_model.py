import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(CNNModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Get input dimensions
        self.in_channels = input_shape[0]  # number of features
        self.sequence_length = input_shape[1]  # sequence length
        
        # Calculate padding to maintain sequence length
        kernel_size = 3
        padding = kernel_size // 2
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv1d(in_channels=self.in_channels, out_channels=32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            
            # Second convolutional block
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            
            # Third convolutional block
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_conv_output_size()
        
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize optimizer and loss function
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _get_conv_output_size(self):
        # Create a dummy input to calculate the size of flattened features
        x = torch.randn(1, self.in_channels, self.sequence_length)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)
        
    def forward(self, x):
        # Ensure input is the right shape (batch_size, channels, sequence_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense_layers(x)
        return x
        
    def compile_model(self, learning_rate=0.001):
        """Set up optimizer and loss function"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)  # LongTensor for class indices
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)  # LongTensor for class indices
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model weights
                best_weights = self.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping triggered')
                    # Restore best weights
                    self.load_state_dict(best_weights)
                    break
        
        return history
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate the model on test data"""
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self(batch_X)
                loss = self.criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                _, true_classes = torch.max(batch_y, 1)
                total += true_classes.size(0)
                correct += (predicted == true_classes).sum().item()
        
        return test_loss / len(test_loader), correct / total
    
    def predict(self, X):
        """Make predictions on new data"""
        X = torch.FloatTensor(X).to(self.device)
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def save_model(self, filepath):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }, filepath)
    
    def load_model(self, filepath):
        """Load a model from a file"""
        checkpoint = torch.load(filepath)
        self.input_shape = checkpoint['input_shape']
        self.num_classes = checkpoint['num_classes']
        self.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 