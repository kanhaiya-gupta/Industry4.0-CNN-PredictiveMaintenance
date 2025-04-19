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
    
    # Check if we have data for both classes
    has_non_fault = np.any(labels == 0)
    has_fault = np.any(labels == 1)
    
    if has_non_fault:
        sns.kdeplot(data=similarity_scores[labels == 0], label='Non-Fault', color=colors[0], fill=True, warn_singular=False)
    if has_fault:
        sns.kdeplot(data=similarity_scores[labels == 1], label='Fault', color=colors[1], fill=True, warn_singular=False)
    
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
            if isinstance(metrics, dict):
                for model_name, model_metrics in metrics.items():
                    f.write(f"\n{model_name.upper()} Model:\n")
                    if isinstance(model_metrics, dict):
                        for metric, value in model_metrics.items():
                            if isinstance(value, (int, float)):
                                f.write(f"{metric}: {value:.4f}\n")
                            else:
                                f.write(f"{metric}: {value}\n")
                    else:
                        f.write(f"Overall: {model_metrics:.4f}\n")
            else:
                f.write(f"Overall: {metrics:.4f}\n")
        
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
        
        # Get features list from DataProcessor
        features = data_processor.features
        logger.info(f"Using features: {features}")
        
        # Visualize data
        data_processor.visualize_data(df, features, results_dir / 'data_visualization')
        
        processed_data = data_processor.preprocess_data(df)
        logger.info(f"Processed data type: {type(processed_data)}")
        logger.info(f"Processed data length: {len(processed_data)}")
        
        try:
            train_data, test_data = processed_data
            logger.info(f"Successfully unpacked train_data and test_data")
        except ValueError as e:
            logger.error(f"Error unpacking processed_data: {str(e)}")
            logger.error(f"Processed data structure: {type(processed_data)}")
            if hasattr(processed_data, '__len__'):
                logger.error(f"Processed data length: {len(processed_data)}")
            raise
        
        logger.info(f"Train data type: {type(train_data)}")
        logger.info(f"Test data type: {type(test_data)}")
        
        try:
            X_train, Y_train = train_data
            logger.info(f"Successfully unpacked X_train and Y_train")
        except ValueError as e:
            logger.error(f"Error unpacking train_data: {str(e)}")
            logger.error(f"Train data structure: {type(train_data)}")
            if hasattr(train_data, '__len__'):
                logger.error(f"Train data length: {len(train_data)}")
            raise
            
        try:
            X_test, Y_test = test_data
            logger.info(f"Successfully unpacked X_test and Y_test")
        except ValueError as e:
            logger.error(f"Error unpacking test_data: {str(e)}")
            logger.error(f"Test data structure: {type(test_data)}")
            if hasattr(test_data, '__len__'):
                logger.error(f"Test data length: {len(test_data)}")
            raise
            
        logger.info("Data loaded and preprocessed")

        # Train CNN Model
        logger.info("Training CNN model...")
        cnn_model = CNNModel(
            input_shape=(8, 18),  # sequence_length=8, num_features=18
            num_classes=config['model']['cnn']['num_classes'],
            config=config,
            debug_mode=False
        )
        cnn_model.compile_model(learning_rate=config['model']['cnn']['learning_rate'])
        
        # Create DataLoader for CNN training
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(Y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['cnn']['batch_size'],
            shuffle=True
        )
        
        # Create validation DataLoader
        val_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(Y_test)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['model']['cnn']['batch_size'],
            shuffle=False
        )
        
        # Train CNN model
        cnn_history = cnn_model.train_model(
            train_loader,
            val_loader,
            epochs=config['model']['cnn']['epochs']
        )
        logger.info("CNN model training completed")

        # Save CNN model
        cnn_model_path = Path(config['paths']['models_dir']) / config['paths']['cnn_model_file']
        try:
            cnn_model.save_model(cnn_model_path)
            if not cnn_model_path.exists():
                raise FileNotFoundError(f"Failed to save CNN model to {cnn_model_path}")
            logger.info(f"CNN model saved successfully to {cnn_model_path}")
            
            # Verify model can be loaded
            test_model = CNNModel()
            test_model.load_model(cnn_model_path)
            logger.info("CNN model verification successful")
        except Exception as e:
            logger.error(f"Error saving CNN model: {str(e)}")
            raise

        # Evaluate CNN model
        cnn_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(cnn_model.device)
            Y_test_tensor = torch.LongTensor(Y_test).to(cnn_model.device)
            outputs = cnn_model(X_test_tensor)
            cnn_probs = torch.softmax(outputs, dim=1)
            _, cnn_pred = torch.max(outputs, 1)
            cnn_metrics = evaluate_model(Y_test, cnn_pred.cpu().numpy(), config)
            logger.info("CNN Evaluation metrics:")
            for metric, value in cnn_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

        # Save CNN metrics
        cnn_metrics_file = results_dir / 'cnn_metrics.txt'
        save_metrics(cnn_metrics, cnn_metrics_file)

        # Plot CNN classifier results
        cnn_scores = cnn_probs[:, 1].cpu().numpy()  # Probability of fault class
        plot_classifier_results(cnn_scores, Y_test, config)
        cnn_classifier_file = results_dir / 'cnn_classifier_results.png'
        save_plot(plt.gcf(), cnn_classifier_file, "CNN Classifier Results")
        plt.close()

        # Plot CNN confusion matrix
        cnn_cm = confusion_matrix(Y_test, cnn_pred.cpu().numpy())
        plot_confusion_matrix(cnn_cm, ['Non-Fault', 'Fault'], title='CNN Confusion Matrix', config=config)
        cnn_confusion_file = results_dir / 'cnn_confusion_matrix.png'
        save_plot(plt.gcf(), cnn_confusion_file, "CNN Confusion Matrix")
        plt.close()

        # Plot CNN training history
        plt.figure(figsize=config['visualization']['figure_size'])
        plt.subplot(1, 2, 1)
        plt.plot(cnn_history['train_loss'], label='Training Loss')
        plt.plot(cnn_history['val_loss'], label='Validation Loss')
        plt.title('CNN Loss History', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Epoch', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Loss', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(cnn_history['train_acc'], label='Training Accuracy')
        plt.plot(cnn_history['val_acc'], label='Validation Accuracy')
        plt.title('CNN Accuracy History', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Epoch', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Accuracy', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        cnn_history_file = results_dir / 'cnn_training_history.png'
        save_plot(plt.gcf(), cnn_history_file, "CNN Training History")
        plt.close()

        # Plot CNN accuracy distribution
        plt.figure(figsize=config['visualization']['figure_size'])
        plt.hist(cnn_history['train_acc'], bins=20, alpha=0.5, label='Training Accuracy')
        plt.hist(cnn_history['val_acc'], bins=20, alpha=0.5, label='Validation Accuracy')
        plt.title('CNN Accuracy Distribution', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Accuracy', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Frequency', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        cnn_acc_file = results_dir / 'cnn_accuracy_distribution.png'
        save_plot(plt.gcf(), cnn_acc_file, "CNN Accuracy Distribution")
        plt.close()

        # Plot CNN ROC curve
        cnn_roc_path = results_dir / 'cnn_roc_curves.png'
        cnn_model.plot_roc_curve(train_loader, val_loader, cnn_roc_path)
        logger.info(f"CNN ROC curves saved to {cnn_roc_path}")

        # Generate Siamese pairs and train Siamese model
        logger.info("Training Siamese model...")
        try:
            left_input, right_input, targets = data_processor.generate_siamese_pairs(X_train, Y_train)
            logger.info(f"Successfully generated Siamese pairs: left_input shape={left_input.shape}, right_input shape={right_input.shape}, targets shape={targets.shape}")
        except ValueError as e:
            logger.error(f"Error generating Siamese pairs: {str(e)}")
            raise

        # Split into train and validation sets
        val_size = int(0.2 * len(left_input))
        train_size = len(left_input) - val_size
        
        # Create train dataset
        train_dataset = TensorDataset(
            torch.FloatTensor(left_input[:train_size]),
            torch.FloatTensor(right_input[:train_size]),
            torch.FloatTensor(targets[:train_size])
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['siamese']['batch_size'],
            shuffle=True
        )
        
        # Create validation dataset
        val_dataset = TensorDataset(
            torch.FloatTensor(left_input[train_size:]),
            torch.FloatTensor(right_input[train_size:]),
            torch.FloatTensor(targets[train_size:])
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['model']['siamese']['batch_size'],
            shuffle=False
        )
        
        # Initialize and train Siamese model
        siamese_model = SiameseNetwork(input_shape=(8, 18), debug_mode=False)  # sequence_length=8, num_features=18
        siamese_model.compile_model(learning_rate=config['model']['siamese']['learning_rate'])
        
        siamese_history = siamese_model.train_model(train_loader, val_loader, epochs=config['model']['siamese']['epochs'])
        logger.info("Siamese model training completed")

        # Save the Siamese model
        siamese_model_path = Path(config['paths']['models_dir']) / config['paths']['siamese_model_file']
        try:
            siamese_model.save_model(siamese_model_path)
            if not siamese_model_path.exists():
                raise FileNotFoundError(f"Failed to save Siamese model to {siamese_model_path}")
            logger.info(f"Siamese model saved successfully to {siamese_model_path}")
            
            # Verify model can be loaded
            test_model = SiameseNetwork(input_shape=(8, 18))
            test_model.load_model(siamese_model_path)
            logger.info("Siamese model verification successful")
        except Exception as e:
            logger.error(f"Error saving Siamese model: {str(e)}")
            raise

        # Evaluate the Siamese model
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

        # Calculate Siamese metrics
        siamese_metrics = evaluate_model(real_labels, pred_labels, config)
        logger.info("Siamese Evaluation metrics:")
        for metric, value in siamese_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save Siamese metrics
        siamese_metrics_file = results_dir / 'siamese_metrics.txt'
        save_metrics(siamese_metrics, siamese_metrics_file)

        # Plot Siamese classifier results
        plot_classifier_results(similarity_scores, real_labels, config)
        siamese_classifier_file = results_dir / 'siamese_classifier_results.png'
        save_plot(plt.gcf(), siamese_classifier_file, "Siamese Classifier Results")
        plt.close()

        # Plot Siamese confusion matrix
        siamese_cm = confusion_matrix(real_labels, pred_labels)
        plot_confusion_matrix(siamese_cm, ['Non-Fault', 'Fault'], title='Siamese Confusion Matrix', config=config)
        siamese_confusion_file = results_dir / 'siamese_confusion_matrix.png'
        save_plot(plt.gcf(), siamese_confusion_file, "Siamese Confusion Matrix")
        plt.close()

        # Plot Siamese training history
        plt.figure(figsize=config['visualization']['figure_size'])
        plt.subplot(1, 2, 1)
        plt.plot(siamese_history['train_loss'], label='Training Loss')
        plt.plot(siamese_history['val_loss'], label='Validation Loss')
        plt.title('Siamese Loss History', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Epoch', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Loss', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(siamese_history['train_acc'], label='Training Accuracy')
        plt.plot(siamese_history['val_acc'], label='Validation Accuracy')
        plt.title('Siamese Accuracy History', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Epoch', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Accuracy', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        siamese_history_file = results_dir / 'siamese_training_history.png'
        save_plot(plt.gcf(), siamese_history_file, "Siamese Training History")
        plt.close()

        # Plot Siamese accuracy distribution
        plt.figure(figsize=config['visualization']['figure_size'])
        plt.hist(siamese_history['train_acc'], bins=20, alpha=0.5, label='Training Accuracy')
        plt.hist(siamese_history['val_acc'], bins=20, alpha=0.5, label='Validation Accuracy')
        plt.title('Siamese Accuracy Distribution', fontsize=config['visualization']['font_size']['title'])
        plt.xlabel('Accuracy', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Frequency', fontsize=config['visualization']['font_size']['label'])
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        siamese_acc_file = results_dir / 'siamese_accuracy_distribution.png'
        save_plot(plt.gcf(), siamese_acc_file, "Siamese Accuracy Distribution")
        plt.close()

        # Plot ROC curves for Siamese model
        siamese_roc_path = results_dir / 'siamese_roc_curves.png'
        siamese_model.plot_roc_curve(train_loader, val_loader, siamese_roc_path)
        logger.info(f"Siamese ROC curves saved to {siamese_roc_path}")

        # Save combined metrics
        metrics = {
            'cnn': cnn_metrics,
            'siamese': siamese_metrics
        }
        metrics_file = results_dir / 'combined_metrics.txt'
        save_metrics(metrics, metrics_file)
        logger.info(f"Successfully saved combined metrics to {metrics_file}")

        # Compare models
        logger.info("\nModel Comparison:")
        logger.info("=" * 50)
        logger.info("Metric\t\tCNN\t\tSiamese\t\tDifference")
        logger.info("-" * 50)
        for metric in config['evaluation']['metrics']:
            cnn_value = cnn_metrics[metric]
            siamese_value = siamese_metrics[metric]
            diff = cnn_value - siamese_value
            logger.info(f"{metric}\t\t{cnn_value:.4f}\t\t{siamese_value:.4f}\t\t{diff:+.4f}")
        logger.info("=" * 50)

        # Plot comparison
        plt.figure(figsize=config['visualization']['figure_size'])
        metrics_list = list(cnn_metrics.keys())
        x = np.arange(len(metrics_list))
        width = 0.35

        plt.bar(x - width/2, [cnn_metrics[m] for m in metrics_list], width, label='CNN')
        plt.bar(x + width/2, [siamese_metrics[m] for m in metrics_list], width, label='Siamese')

        plt.xlabel('Metrics', fontsize=config['visualization']['font_size']['label'])
        plt.ylabel('Score', fontsize=config['visualization']['font_size']['label'])
        plt.title('Model Comparison', fontsize=config['visualization']['font_size']['title'])
        plt.xticks(x, metrics_list)
        plt.legend(fontsize=config['visualization']['font_size']['label'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save comparison plot
        comparison_file = results_dir / 'model_comparison.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Successfully saved model comparison to {comparison_file}")

        return {'status': 'success', 'metrics': metrics}

    except Exception as e:
        logger.error(f"Error in training and evaluation: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Error traceback: {e.__traceback__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    train_and_evaluate()
