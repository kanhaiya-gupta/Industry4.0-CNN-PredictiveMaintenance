import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model
import seaborn as sns
from .evaluation import ModelEvaluator

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_tsne_embeddings(embeddings, labels, class_names):
    """
    Plot t-SNE visualization of embeddings with different colors for each class.
    
    Args:
        embeddings: numpy array of embeddings
        labels: numpy array of true labels
        class_names: list of class names
    """
    # Create color map
    colors = {i: plt.cm.Set1(i) for i in range(len(class_names))}
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, n_iter=300, perplexity=5)
    T = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(20,12))
    for i in range(len(class_names)):
        mask = np.argmax(labels, axis=1) == i
        plt.scatter(T[mask, 0], T[mask, 1], c=[colors[i]], label=class_names[i])
    
    plt.legend()
    plt.title('t-SNE Visualization of Embeddings')
    plt.show()

class ModelVisualizer:
    def __init__(self, model):
        self.model = model
        
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        if 'val_acc' in history:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('results/training_history.png')
        plt.close()
        
    def plot_evaluation_metrics(self, metrics, model_name):
        """Plot evaluation metrics"""
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{model_name.lower()}.png')
        plt.close()
        
        # Print classification report
        print(f"\nClassification Report - {model_name}:")
        print(metrics['classification_report'])
        
    def plot_tsne_embeddings(self, X, y, layer_name='global_average_pooling1d'):
        """Plot t-SNE visualization of model embeddings"""
        # Get embeddings from specified layer
        if isinstance(self.model, Model):
            emb_model = Model(inputs=self.model.input, 
                            outputs=self.model.get_layer(layer_name).output)
            embeddings = emb_model.predict(X)
        else:
            embeddings = X
            
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Model Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()
        
    def plot_feature_importance(self, feature_names, importance_scores):
        """Plot feature importance scores"""
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importance_scores)
        plt.barh(range(len(importance_scores)), importance_scores[sorted_idx])
        plt.yticks(range(len(importance_scores)), [feature_names[i] for i in sorted_idx])
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
    def plot_prediction_distribution(self, y_true, y_pred):
        """Plot distribution of predictions vs true labels"""
        plt.figure(figsize=(10, 6))
        plt.hist([y_true, y_pred], bins=20, alpha=0.5, label=['True Labels', 'Predictions'])
        plt.title('Distribution of Predictions vs True Labels')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.legend()
        plt.show()
        
    def compare_models(self, models, X_test, y_test, model_names):
        """Compare performance of multiple models"""
        plt.figure(figsize=(12, 6))
        
        for model, name in zip(models, model_names):
            # Create a temporary evaluator for each model
            evaluator = ModelEvaluator(model, None)
            metrics = evaluator.evaluate_model(X_test, y_test, 
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
    
    def plot_roc_curve(self, fpr, tpr, auc):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_precision_recall_curve(self, precision, recall, average_precision):
        """Plot precision-recall curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show() 