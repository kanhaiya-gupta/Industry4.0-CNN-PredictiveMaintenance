import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from .data_processor import DataProcessor
import seaborn as sns
import torch

class ModelEvaluator:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        
    def evaluate_model(self, X_test, y_test, model_type='cnn'):
        """Evaluate model performance"""
        if model_type == 'cnn':
            # Convert to PyTorch tensors
            X_test = torch.FloatTensor(X_test)
            y_test = torch.LongTensor(y_test)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test)
                _, predictions = torch.max(outputs, 1)
                predictions = predictions.numpy()
                y_true = y_test.numpy()
        else:  # siamese
            # Convert to PyTorch tensors
            X_test = torch.FloatTensor(X_test)
            y_test = torch.FloatTensor(y_test)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_test, X_test)
                predictions = (outputs > 0.5).float().squeeze().numpy()  # Remove extra dimension
                y_true = y_test.numpy()
            
        # Get unique classes in the data
        unique_classes = np.unique(np.concatenate([y_true, predictions]))
        
        metrics = {
            'classification_report': classification_report(y_true, predictions, 
                                                         labels=unique_classes,
                                                         zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, predictions, 
                                               labels=unique_classes)
        }
        
        return metrics
        
    def plot_evaluation_metrics(self, metrics, class_names):
        """Plot evaluation metrics including confusion matrix"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png')
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
    def compare_models(self, models, X_test, y_test, model_names):
        """Compare performance of multiple models"""
        plt.figure(figsize=(12, 6))
        
        for model, name in zip(models, model_names):
            metrics = self.evaluate_model(X_test, y_test, 
                                       model_type='siamese' if 'siamese' in name.lower() else 'cnn')
            cm = metrics['confusion_matrix']
            accuracy = np.trace(cm) / np.sum(cm)
            plt.bar(name, accuracy)
            
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('results/model_comparison.png')
        plt.close()
    
    def evaluate_with_different_thresholds(self, X_test, y_test, thresholds):
        """Evaluate model with different decision thresholds"""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model must have predict_proba method")
            
        y_proba = self.model.predict_proba(X_test)[:, 1]
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int)
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            results.append((threshold, accuracy))
            
        # Plot threshold vs accuracy
        thresholds, accuracies = zip(*results)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies)
        plt.title('Accuracy vs Decision Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()
        
        return results 