import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SiameseNetwork(nn.Module):
    def __init__(self, input_shape):
        super(SiameseNetwork, self).__init__()
        self.input_shape = input_shape
        
        # Get input dimensions
        self.sequence_length = input_shape[0]  # sequence length (8)
        self.in_channels = input_shape[1]  # number of features (18)
        
        # Calculate padding to maintain sequence length
        kernel_size = 3
        padding = kernel_size // 2
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv1d(in_channels=self.in_channels, out_channels=64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            
            # Second convolutional block
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),
            
            # Third convolutional block
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_flattened_size()
        
        # Dense layers for embedding
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128)
        )
        
        # Distance network with sigmoid activation
        self.distance_net = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()  # Add sigmoid activation to ensure output is between 0 and 1
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def _get_flattened_size(self):
        # Create a dummy input to calculate the size after convolutions
        x = torch.randn(1, self.in_channels, self.sequence_length)
        x = self.encoder(x)
        return x.numel() // x.size(0)
        
    def forward_one(self, x):
        # Ensure input is the right shape (batch_size, channels, sequence_length)
        if len(x.shape) == 2:
            # Input is [batch_size, features] -> [batch_size, features, 1]
            x = x.unsqueeze(-1)
        elif len(x.shape) == 3:
            # Input is [batch_size, sequence_length, features] -> [batch_size, features, sequence_length]
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 4:
            # Input is [batch_size, pairs, sequence_length, features] -> [batch_size * pairs, features, sequence_length]
            x = x.view(-1, x.size(2), x.size(3))
            x = x.permute(0, 2, 1)
            
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.embedding(x)
        return x
        
    def forward(self, x1, x2=None):
        if x2 is None:
            # Single input mode for feature extraction
            return self.forward_one(x1)
            
        # Siamese mode
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        
        # Calculate Euclidean distance
        distance = torch.pairwise_distance(output1, output2, keepdim=True)
        
        # Pass through distance network (sigmoid is already applied in the network)
        similarity = self.distance_net(distance)
        
        return similarity
        
    def compile_model(self, learning_rate=0.001):
        """Set up optimizer and loss function"""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def train_model(self, train_loader, epochs=50):
        """Train the model using a DataLoader"""
        # Training loop
        history = {'train_loss': [], 'train_acc': []}
        
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X1, batch_X2, batch_labels in train_loader:
                # Move data to device
                batch_X1 = batch_X1.to(self.device)
                batch_X2 = batch_X2.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self(batch_X1, batch_X2)  # outputs are already between 0 and 1
                loss = self.criterion(outputs, batch_labels.unsqueeze(1).float())
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs.data >= 0.5).float()
                train_total += batch_labels.size(0)
                train_correct += (predicted.squeeze() == batch_labels).sum().item()
            
            # Calculate epoch metrics
            train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        return history
    
    def evaluate(self, X_test, batch_size=32):
        """Extract features for evaluation"""
        X_test = torch.FloatTensor(X_test).to(self.device)
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.eval()
        features = []
        
        with torch.no_grad():
            for (batch_X,) in test_loader:
                batch_features = self.forward_one(batch_X)
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)
    
    def save_model(self, filepath):
        """Save the model to a file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'input_shape': self.input_shape
        }, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            checkpoint = torch.load(filepath, weights_only=True)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    input_shape = checkpoint.get('input_shape', self.input_shape)
                else:
                    state_dict = checkpoint
                    input_shape = (19, 1)  # Based on the saved model's encoder layer
            else:
                state_dict = checkpoint
                input_shape = (19, 1)
            
            # Reinitialize the model with the correct input shape
            self.__init__(input_shape=input_shape)
            self.load_state_dict(state_dict)
            self.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}") 