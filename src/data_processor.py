import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import librosa
import tensorflow as tf

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, data_type='sensor'):
        """Load and preprocess the data based on type"""
        if data_type == 'sensor':
            data = pd.read_csv(self.data_path)
            # Assuming the last column is the label
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        elif data_type == 'audio':
            data, _ = librosa.load(self.data_path)
            X = data.reshape(-1, 1)
            y = None  # Audio data might need different label handling
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        print(f"Data shape: {X.shape}")
        return X, y
        
    def preprocess_data(self, data, labels=None, data_type='sensor', test_size=0.2, random_state=42):
        """Preprocess the data including scaling and train-test split"""
        X, y = data if isinstance(data, tuple) else (data, labels)
        
        if data_type == 'sensor':
            # Scale the features
            X = self.scaler.fit_transform(X)
            
            # For PyTorch Conv1D, shape should be (batch_size, channels, sequence_length)
            # Here we treat each feature as a channel and use sequence_length of 1
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Encode labels if provided
            if y is not None:
                # For PyTorch CrossEntropyLoss, we need class indices, not one-hot encoding
                y = self.label_encoder.fit_transform(y)
            
        elif data_type == 'audio':
            # Extract features using librosa
            mfccs = librosa.feature.mfcc(y=X.flatten(), sr=22050, n_mfcc=13)
            X = mfccs.T
            
            # Scale the features
            X = self.scaler.fit_transform(X)
            
            # Reshape for CNN input (batch_size, channels, sequence_length)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Split into train and test sets
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            return X_train, X_test
    
    def create_siamese_pairs(self, X, y):
        """Create pairs of samples for Siamese network training."""
        num_samples = len(X)
        pairs = []
        labels = []
        
        # Create positive pairs (same class)
        for i in range(num_samples):
            for j in range(i + 1, min(i + 2, num_samples)):  # Limit pairs per sample
                if y[i] == y[j]:
                    pairs.append([X[i], X[j]])
                    labels.append(1.0)  # Similar pair
        
        # Create negative pairs (different classes)
        for i in range(num_samples):
            for j in range(i + 1, min(i + 2, num_samples)):
                if y[i] != y[j]:
                    pairs.append([X[i], X[j]])
                    labels.append(0.0)  # Dissimilar pair
        
        # Convert to numpy arrays
        pairs = np.array(pairs)
        labels = np.array(labels)
        
        # Shuffle pairs and labels together
        indices = np.random.permutation(len(pairs))
        pairs = pairs[indices]
        labels = labels[indices]
        
        return pairs, labels
    
    def plot_confusion_matrix(self, cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        """Plot confusion matrix"""
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=25)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)
        
        fmt = '.2f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)
        
        plt.ylabel('True label', fontsize=20)
        plt.xlabel('Predicted label', fontsize=20)
        plt.tight_layout()
        return plt 