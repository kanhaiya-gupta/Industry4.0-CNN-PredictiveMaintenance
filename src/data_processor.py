import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import librosa
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from src.utils.logger import setup_logging
import seaborn as sns

logger = setup_logging('data_processor')

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.features = [
            'I_w_BLO_Weg', 'O_w_BLO_power', 'O_w_BLO_voltage', 'I_w_BHL_Weg', 
            'O_w_BHL_power', 'O_w_BHL_voltage', 'I_w_BHR_Weg', 'O_w_BHR_power', 
            'O_w_BHR_voltage', 'I_w_BRU_Weg', 'O_w_BRU_power', 'O_w_BRU_voltage', 
            'I_w_HR_Weg', 'O_w_HR_power', 'O_w_HR_voltage', 'I_w_HL_Weg', 
            'O_w_HL_power', 'O_w_HL_voltage'
        ]
        self.sequence_length = 8
        self.test_size = 0.1
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, data_path):
        """Load and preprocess the data."""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df):
        """Preprocess the data including sequence generation and normalization."""
        try:
            # Convert timestamp to cycles
            k = -1
            cycle = []
            for time in df.Timestamp:
                if time == 0.:
                    k += 1
                    cycle.append(k)
                else:
                    cycle.append(k)
            df['Timestamp'] = cycle

            # Generate sequences
            X_train, X_test = [], []
            Y_train, Y_test = [], []

            for k, group_df in df.groupby('Timestamp'):
                if int(len(group_df) * self.test_size) > self.sequence_length:
                    init = group_df[self.features].values[0] + 1e-3
                    y = group_df.Labels.values
                    x_train = group_df[self.features].values[:int(len(group_df) * (1 - self.test_size))] / init
                    x_test = group_df[self.features].values[int(len(group_df) * (1 - self.test_size)):] / init

                    x_train = self.scaler.fit_transform(x_train)
                    x_test = self.scaler.transform(x_test)

                    x_train = self._gen_sequence(x_train, self.sequence_length)
                    x_test = self._gen_sequence(x_test, self.sequence_length)
                    y_train = y[self.sequence_length:int(len(group_df) * (1 - self.test_size))]
                    y_test = y[int(len(group_df) * (1 - self.test_size)) + self.sequence_length:]

                    Y_train.append(y_train)
                    Y_test.append(y_test)
                    X_train.append(x_train)
                    X_test.append(x_test)

            X_train, X_test = np.vstack(X_train), np.vstack(X_test)
            Y_train, Y_test = np.concatenate(Y_train), np.concatenate(Y_test)

            logger.info(f"Generated sequences - Train: {X_train.shape}, Test: {X_test.shape}")
            result = ((X_train, Y_train), (X_test, Y_test))
            logger.info(f"Returning data structure: {type(result)} with length {len(result)}")
            logger.info(f"First element type: {type(result[0])} with length {len(result[0])}")
            logger.info(f"Second element type: {type(result[1])} with length {len(result[1])}")
            return result
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def _gen_sequence(self, df, seq_length):
        """Generate sequences from the data."""
        seq_df = []
        for start, stop in zip(range(0, len(df) - seq_length), range(seq_length, len(df))):
            seq_df.append(df[start:stop, :])
        return np.asarray(seq_df)

    def generate_siamese_pairs(self, X_train, Y_train, pairs=3):
        """Generate pairs for Siamese network training."""
        left_input = []
        right_input = []
        targets = []

        for i in range(len(Y_train)):
            for _ in range(pairs):
                compare_to = i
                while compare_to == i:
                    compare_to = np.random.randint(0, len(Y_train) - 1)
                left_input.append(X_train[i])
                right_input.append(X_train[compare_to])
                targets.append(1. if Y_train[i] == Y_train[compare_to] else 0.)

        left_input = np.asarray(left_input).reshape(-1, self.sequence_length, len(self.features))
        right_input = np.asarray(right_input).reshape(-1, self.sequence_length, len(self.features))
        targets = np.asarray(targets)

        logger.info(f"Generated Siamese pairs - Left: {left_input.shape}, Right: {right_input.shape}")
        return left_input, right_input, targets

    def create_siamese_pairs(self, X, y):
        """Create pairs of samples for Siamese network training."""
        left_input = []
        right_input = []
        targets = []
        
        # Create positive pairs (same class)
        for i in range(len(X)):
            for j in range(i + 1, min(i + 2, len(X))):  # Limit pairs per sample
                if y[i] == y[j]:
                    left_input.append(X[i])
                    right_input.append(X[j])
                    targets.append(1.0)  # Similar pair
        
        # Create negative pairs (different classes)
        for i in range(len(X)):
            for j in range(i + 1, min(i + 2, len(X))):
                if y[i] != y[j]:
                    left_input.append(X[i])
                    right_input.append(X[j])
                    targets.append(0.0)  # Dissimilar pair
        
        # Convert to numpy arrays
        left_input = np.asarray(left_input).reshape(-1, self.sequence_length, len(self.features))
        right_input = np.asarray(right_input).reshape(-1, self.sequence_length, len(self.features))
        targets = np.asarray(targets)
        
        # Shuffle pairs and labels together
        indices = np.random.permutation(len(targets))
        left_input = left_input[indices]
        right_input = right_input[indices]
        targets = targets[indices]
        
        logger.info(f"Created Siamese pairs - Left: {left_input.shape}, Right: {right_input.shape}")
        return left_input, right_input, targets
    
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

    def visualize_data(self, df, features, save_dir=None):
        """Visualize data distribution and patterns"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Time series of Timestamp and Labels
        ax1 = plt.subplot(2, 1, 1)
        df.Timestamp.plot(ax=ax1, figsize=(16,6), color='#1f77b4', linewidth=2)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Timestamp', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(df.Labels, color='#ff7f0e', linewidth=3, alpha=0.7)
        ax2.set_ylabel('Failure', fontsize=12)
        plt.title('Time Series of Timestamps and Failure Labels', fontsize=14, pad=20)
        
        # Plot 2: Standard deviation of features
        plt.subplot(2, 1, 2)
        df.loc[:,features].std().plot.bar(figsize=(16,6), color='#2ca02c')
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Feature Standard Deviations', fontsize=14, pad=20)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add correlation heatmap
        plt.figure(figsize=(16, 12))
        corr_matrix = df.loc[:,features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Save plots if directory provided
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save time series plot
            plt.figure(figsize=(20, 12))
            ax1 = plt.subplot(2, 1, 1)
            df.Timestamp.plot(ax=ax1, figsize=(16,6), color='#1f77b4', linewidth=2)
            ax1.set_xlabel('Time', fontsize=12)
            ax1.set_ylabel('Timestamp', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            ax2.plot(df.Labels, color='#ff7f0e', linewidth=3, alpha=0.7)
            ax2.set_ylabel('Failure', fontsize=12)
            plt.title('Time Series of Timestamps and Failure Labels', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(save_dir / 'time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save standard deviation plot
            plt.figure(figsize=(16,6))
            df.loc[:,features].std().plot.bar(color='#2ca02c')
            plt.ylabel('Standard Deviation', fontsize=12)
            plt.title('Feature Standard Deviations', fontsize=14, pad=20)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'feature_std.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save correlation heatmap
            plt.figure(figsize=(16, 12))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Feature Correlation Matrix', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(save_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Data visualization plots saved to {save_dir}") 