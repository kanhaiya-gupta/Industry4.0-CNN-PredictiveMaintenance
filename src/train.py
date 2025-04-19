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
import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import itertools
from sklearn.manifold import TSNE
import seaborn as sns

# Initialize logger
logger = setup_logging('training')

def ensure_results_dir(config=None):
    """Ensure results directory exists."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, config=None):
    """Plot confusion matrix using configuration settings."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    # Get visualization settings from config
    fig_size = config['visualization']['figure_size']
    title_font = config['visualization']['font_size']['title']
    label_font = config['visualization']['font_size']['label']
    tick_font = config['visualization']['font_size']['tick']
    
    plt.figure(figsize=fig_size)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=title_font)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=tick_font)
    plt.yticks(tick_marks, classes, fontsize=tick_font)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", 
                 fontsize=tick_font)

    plt.ylabel('True label', fontsize=label_font)
    plt.xlabel('Predicted label', fontsize=label_font)
    plt.tight_layout()

def plot_tsne(embeddings, labels, config=None):
    """Plot TSNE visualization of embeddings."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    tsne_config = config['visualization']['tsne']
    colors = config['visualization']['colors']
    
    # Perform TSNE
    tsne = TSNE(
        n_components=tsne_config['n_components'],
        random_state=tsne_config['random_state'],
        perplexity=tsne_config['perplexity'],
        n_iter=tsne_config['n_iter']
    )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=config['visualization']['figure_size'])
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            c=colors[i], 
            label=f'Class {label}',
            alpha=0.6
        )
    
    plt.title('TSNE Visualization of Embeddings', fontsize=config['visualization']['font_size']['title'])
    plt.legend(fontsize=config['visualization']['font_size']['label'])
    plt.tight_layout()

def plot_classifier_results(similarity_scores, labels, config=None):
    """Plot classifier results showing similarity score distributions for fault and non-fault cases."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    # Get visualization settings from config
    fig_size = config['visualization']['figure_size']
    title_font = config['visualization']['font_size']['title']
    label_font = config['visualization']['font_size']['label']
    tick_font = config['visualization']['font_size']['tick']
    colors = config['visualization']['colors']
    
    # Create figure
    plt.figure(figsize=fig_size)
    
    # Plot distributions
    sns.kdeplot(data=similarity_scores[labels == 0], label='Non-Fault', color=colors[0], fill=True)
    sns.kdeplot(data=similarity_scores[labels == 1], label='Fault', color=colors[1], fill=True)
    
    # Add decision threshold
    threshold = config['evaluation']['decision_threshold']
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'Decision Threshold ({threshold})')
    
    # Customize plot
    plt.title('Similarity Score Distributions', fontsize=title_font)
    plt.xlabel('Similarity Score', fontsize=label_font)
    plt.ylabel('Density', fontsize=label_font)
    plt.legend(fontsize=tick_font)
    plt.grid(True, alpha=0.3)
    
    # Add accuracy information
    predictions = (similarity_scores > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    plt.text(0.05, 0.95, f'Accuracy: {accuracy:.4f}', 
             transform=plt.gca().transAxes, 
             fontsize=tick_font,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()

def evaluate_model(y_true, y_pred, config=None):
    """Evaluate model using configured metrics."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    metrics = {}
    for metric in config['evaluation']['metrics']:
        if metric == 'accuracy':
            metrics[metric] = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            metrics[metric] = precision_score(y_true, y_pred, average='weighted')
        elif metric == 'recall':
            metrics[metric] = recall_score(y_true, y_pred, average='weighted')
        elif metric == 'f1_score':
            metrics[metric] = f1_score(y_true, y_pred, average='weighted')
    
    return metrics

def save_plot(fig, filepath, title):
    """Save a plot and verify it was created."""
    try:
        fig.savefig(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Failed to save {title} to {filepath}")
        logger.info(f"Successfully saved {title} to {filepath}")
    except Exception as e:
        logger.error(f"Error saving {title} to {filepath}: {str(e)}")
        raise

def save_metrics(metrics, filepath):
    """Save metrics to file and verify it was created."""
    try:
        with open(filepath, 'w') as f:
            f.write("Evaluation Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Failed to save metrics to {filepath}")
        logger.info(f"Successfully saved metrics to {filepath}")
    except Exception as e:
        logger.error(f"Error saving metrics to {filepath}: {str(e)}")
        raise

def train_and_evaluate():
    """Main training and evaluation function."""
    try:
        # Load configuration
        config = ConfigLoader().get_full_config()
        logger.info("Configuration loaded successfully")

        # Ensure results directory exists
        results_dir = ensure_results_dir(config)
        logger.info(f"Results will be saved in {results_dir}")

        # Initialize data processor
        data_processor = DataProcessor(config)
        logger.info("Data processor initialized")

        # Load and preprocess data
        data_path = Path(config['paths']['data_dir']) / config['paths']['sensor_data_file']
        df = data_processor.load_data(data_path)
        logger.info(f"Loaded data shape: {df.shape}")
        
        processed_data = data_processor.preprocess_data(df)
        logger.info(f"Processed data type: {type(processed_data)}")
        logger.info(f"Processed data length: {len(processed_data)}")
        
        train_data, test_data = processed_data
        logger.info(f"Train data type: {type(train_data)}")
        logger.info(f"Test data type: {type(test_data)}")
        
        X_train, Y_train = train_data
        X_test, Y_test = test_data
        logger.info("Data loaded and preprocessed")

        # Generate Siamese pairs
        left_input, right_input, targets = data_processor.generate_siamese_pairs(X_train, Y_train)
        logger.info("Siamese pairs generated")

        # Initialize and train Siamese model
        siamese_model = SiameseNetwork(input_shape=(8, 18))  # sequence_length=8, num_features=18
        siamese_model.compile_model(learning_rate=config['model']['siamese']['learning_rate'])
        
        # Create DataLoader for training
        train_dataset = TensorDataset(
            torch.FloatTensor(left_input),
            torch.FloatTensor(right_input),
            torch.FloatTensor(targets)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['siamese']['batch_size'],
            shuffle=True
        )
        
        history = siamese_model.train_model(train_loader, epochs=config['model']['siamese']['epochs'])
        logger.info("Model training completed")

        # Save the model
        model_path = Path(config['paths']['models_dir']) / config['paths']['siamese_model_file']
        siamese_model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Evaluate the model
        real = []
        pred = []
        similarity_scores = []
        rep = 4

        # Convert test data to tensors and move to device
        X_test_tensor = torch.FloatTensor(X_test).to(siamese_model.device)
        X_train_tensor = torch.FloatTensor(X_train).to(siamese_model.device)
        Y_train_tensor = torch.FloatTensor(Y_train).to(siamese_model.device)
        Y_test_tensor = torch.FloatTensor(Y_test).to(siamese_model.device)

        # Set model to evaluation mode
        siamese_model.eval()
        with torch.no_grad():
            for j, i in enumerate(tqdm.tqdm(Y_test_tensor)):
                test = X_test_tensor[j].unsqueeze(0)  # Add batch dimension

                # Get non-fault samples
                nofault_indices = torch.where(Y_train_tensor == 0)[0]
                nofault_indices = nofault_indices[torch.randperm(len(nofault_indices))[:rep]]
                nofault_samples = X_train_tensor[nofault_indices]
                nofault_sim = torch.max(torch.stack([
                    siamese_model.predict(test, sample.unsqueeze(0)).squeeze()
                    for sample in nofault_samples
                ])).item()

                # Get fault samples
                fault_indices = torch.where(Y_train_tensor == 1)[0]
                fault_indices = fault_indices[torch.randperm(len(fault_indices))[:rep]]
                fault_samples = X_train_tensor[fault_indices]
                fault_sim = torch.max(torch.stack([
                    siamese_model.predict(test, sample.unsqueeze(0)).squeeze()
                    for sample in fault_samples
                ])).item()
                
                # Store similarity scores
                similarity_scores.append(nofault_sim - fault_sim)  # Difference in similarity scores
                
                pred.append('nofault' if nofault_sim > fault_sim else 'fault')
                real.append('nofault' if i.item() == 0 else 'fault')

        # Convert to numpy arrays for plotting and metrics
        similarity_scores = np.array(similarity_scores)
        real_labels = np.array([0 if r == 'nofault' else 1 for r in real])
        pred_labels = np.array([0 if p == 'nofault' else 1 for p in pred])

        # Plot classifier results
        plot_classifier_results(similarity_scores, real_labels, config)
        classifier_results_file = results_dir / 'classifier_results.png'
        save_plot(plt.gcf(), classifier_results_file, "Classifier Results")
        plt.close()

        # Print classification report
        print(classification_report(real, pred))

        # Calculate evaluation metrics
        metrics = evaluate_model(real_labels, pred_labels, config)
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save metrics
        metrics_file = results_dir / 'metrics.txt'
        save_metrics(metrics, metrics_file)
        logger.info(f"Successfully saved metrics to {metrics_file}")

        # Plot and save confusion matrix
        cm = confusion_matrix(real_labels, pred_labels)
        plot_confusion_matrix(cm, ['Non-Fault', 'Fault'], config=config)
        confusion_matrix_file = results_dir / 'confusion_matrix.png'
        save_plot(plt.gcf(), confusion_matrix_file, "Confusion Matrix")
        plt.close()
        logger.info(f"Successfully saved Confusion Matrix to {confusion_matrix_file}")

        return metrics

    except Exception as e:
        logger.error(f"Error in training and evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    train_and_evaluate()
