import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import setup_logging
from src.config_loader import ConfigLoader

logger = setup_logging('feature_analysis')

def analyze_cnn_features(model, data_loader, feature_names, results_dir, config=None):
    """Analyze feature importance for CNN model using SHAP values."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    logger.info("Analyzing CNN feature importance...")
    
    # Get background data (first batch)
    background_data = next(iter(data_loader))[0]
    background_data = background_data.to(model.device)
    logger.debug(f"Initial background data shape: {background_data.shape}")
    
    # Get sequence length from input shape
    sequence_length = background_data.shape[1]
    num_features = background_data.shape[2]
    logger.debug(f"Sequence length: {sequence_length}, Num features: {num_features}")
    
    # Reshape background data to 2D [batch_size, sequence_length * num_features]
    background_data = background_data.view(background_data.size(0), -1)
    logger.debug(f"Reshaped background data shape: {background_data.shape}")
    
    # Create SHAP explainer
    def cnn_predict(x):
        # Reshape input back to 3D [batch_size, sequence_length, num_features]
        x = x.reshape(-1, sequence_length, num_features)
        logger.debug(f"Reshaped input in cnn_predict: {x.shape}")
        x = torch.tensor(x, dtype=torch.float32).to(model.device)
        with torch.no_grad():
            output = model(x).cpu().numpy()
            logger.debug(f"Model output shape: {output.shape}")
            return output
    
    # Initialize explainer with background data
    background_data_np = background_data.cpu().numpy()
    explainer = shap.KernelExplainer(cnn_predict, background_data_np)
    
    # Calculate SHAP values for a subset of data
    test_data = []
    for batch_X, _ in data_loader:
        # Reshape batch to 2D [batch_size, sequence_length * num_features]
        batch_X = batch_X.view(batch_X.size(0), -1)
        logger.debug(f"Batch shape after reshape: {batch_X.shape}")
        test_data.append(batch_X.cpu().numpy())
        if len(test_data) * batch_X.shape[0] >= 100:  # Use first 100 samples
            break
    test_data = np.vstack(test_data)
    logger.debug(f"Final test data shape: {test_data.shape}")
    
    # Get SHAP values
    shap_values = explainer.shap_values(test_data)
    logger.debug(f"Raw SHAP values type: {type(shap_values)}")
    
    # Handle binary classification case
    if isinstance(shap_values, list):
        logger.debug(f"SHAP values list length: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            logger.debug(f"SHAP values[{i}] shape: {sv.shape}")
        # For binary classification, we only need one class's SHAP values
        shap_values = shap_values[1]  # Use positive class SHAP values
    else:
        logger.debug(f"Single SHAP values shape: {shap_values.shape}")
    
    logger.debug(f"Final SHAP values shape: {shap_values.shape}")
    
    # Create expanded feature names for each time step
    expanded_feature_names = []
    for t in range(sequence_length):
        for feature in feature_names:
            expanded_feature_names.append(f"{feature}_t{t}")
    logger.debug(f"Number of expanded feature names: {len(expanded_feature_names)}")
    
    # Verify shapes match
    if shap_values.shape[1] != test_data.shape[1]:
        logger.error(f"Shape mismatch: SHAP values {shap_values.shape} vs test data {test_data.shape}")
        raise ValueError("SHAP values and test data shapes do not match")
    
    if len(expanded_feature_names) != test_data.shape[1]:
        logger.error(f"Feature names mismatch: {len(expanded_feature_names)} vs {test_data.shape[1]}")
        raise ValueError("Number of feature names does not match data dimensions")
    
    # Plot feature importance
    plt.figure(figsize=config['visualization']['figure_size'])
    shap.summary_plot(
        shap_values,
        test_data,
        feature_names=expanded_feature_names,
        show=False
    )
    plt.title('CNN Feature Importance', fontsize=config['visualization']['font_size']['title'])
    plt.tight_layout()
    
    # Save plot
    cnn_feature_file = Path(results_dir) / 'cnn_feature_importance.png'
    plt.savefig(cnn_feature_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"CNN feature importance plot saved to {cnn_feature_file}")
    
    # Calculate and save feature rankings
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Ensure we have scalar values for feature importance
    if isinstance(mean_abs_shap, np.ndarray) and mean_abs_shap.ndim > 0:
        mean_abs_shap = mean_abs_shap.flatten()
    
    feature_importance = dict(zip(expanded_feature_names, mean_abs_shap))
    sorted_features = sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True)
    
    # Save rankings to file
    with open(Path(results_dir) / 'cnn_feature_rankings.txt', 'w') as f:
        f.write("CNN Feature Rankings (by importance):\n")
        f.write("=" * 50 + "\n")
        for feature, importance in sorted_features:
            f.write(f"{feature}: {float(importance):.6f}\n")
    
    return feature_importance

def analyze_siamese_features(model, data_loader, feature_names, results_dir, config=None):
    """Analyze feature importance for Siamese model using SHAP values."""
    if config is None:
        config = ConfigLoader().get_full_config()
    
    logger.info("Analyzing Siamese feature importance...")
    
    # Get background data (first batch)
    background_data = next(iter(data_loader))[0]  # First input
    background_data = background_data.to(model.device)
    logger.debug(f"Initial background data shape: {background_data.shape}")
    
    # Get sequence length from input shape
    sequence_length = background_data.shape[1]
    num_features = background_data.shape[2]
    logger.debug(f"Sequence length: {sequence_length}, Num features: {num_features}")
    
    # Reshape background data to 2D [batch_size, sequence_length * num_features]
    background_data = background_data.view(background_data.size(0), -1)
    logger.debug(f"Reshaped background data shape: {background_data.shape}")
    
    # Create SHAP explainer
    def siamese_predict(x):
        # Reshape input back to 3D [batch_size, sequence_length, num_features]
        x = x.reshape(-1, sequence_length, num_features)
        logger.debug(f"Reshaped input in siamese_predict: {x.shape}")
        x = torch.tensor(x, dtype=torch.float32).to(model.device)
        with torch.no_grad():
            output = model.forward_one(x).cpu().numpy()
            logger.debug(f"Model output shape: {output.shape}")
            return output
    
    # Initialize explainer with background data
    background_data_np = background_data.cpu().numpy()
    explainer = shap.KernelExplainer(siamese_predict, background_data_np)
    
    # Calculate SHAP values for a subset of data
    test_data = []
    for batch_X1, _, _ in data_loader:
        # Reshape batch to 2D [batch_size, sequence_length * num_features]
        batch_X1 = batch_X1.view(batch_X1.size(0), -1)
        logger.debug(f"Batch shape after reshape: {batch_X1.shape}")
        test_data.append(batch_X1.cpu().numpy())
        if len(test_data) * batch_X1.shape[0] >= 100:  # Use first 100 samples
            break
    test_data = np.vstack(test_data)
    logger.debug(f"Final test data shape: {test_data.shape}")
    
    # Get SHAP values
    shap_values = explainer.shap_values(test_data)
    logger.debug(f"Raw SHAP values type: {type(shap_values)}")
    
    # Handle binary classification case
    if isinstance(shap_values, list):
        logger.debug(f"SHAP values list length: {len(shap_values)}")
        for i, sv in enumerate(shap_values):
            logger.debug(f"SHAP values[{i}] shape: {sv.shape}")
        # For binary classification, we only need one class's SHAP values
        shap_values = shap_values[1]  # Use positive class SHAP values
    else:
        logger.debug(f"Single SHAP values shape: {shap_values.shape}")
    
    logger.debug(f"Final SHAP values shape: {shap_values.shape}")
    
    # Create expanded feature names for each time step
    expanded_feature_names = []
    for t in range(sequence_length):
        for feature in feature_names:
            expanded_feature_names.append(f"{feature}_t{t}")
    logger.debug(f"Number of expanded feature names: {len(expanded_feature_names)}")
    
    # Verify shapes match
    if shap_values.shape[1] != test_data.shape[1]:
        logger.error(f"Shape mismatch: SHAP values {shap_values.shape} vs test data {test_data.shape}")
        raise ValueError("SHAP values and test data shapes do not match")
    
    if len(expanded_feature_names) != test_data.shape[1]:
        logger.error(f"Feature names mismatch: {len(expanded_feature_names)} vs {test_data.shape[1]}")
        raise ValueError("Number of feature names does not match data dimensions")
    
    # Plot feature importance
    plt.figure(figsize=config['visualization']['figure_size'])
    shap.summary_plot(
        shap_values,
        test_data,
        feature_names=expanded_feature_names,
        show=False
    )
    plt.title('Siamese Feature Importance', fontsize=config['visualization']['font_size']['title'])
    plt.tight_layout()
    
    # Save plot
    siamese_feature_file = Path(results_dir) / 'siamese_feature_importance.png'
    plt.savefig(siamese_feature_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Siamese feature importance plot saved to {siamese_feature_file}")
    
    # Calculate and save feature rankings
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    # Ensure we have scalar values for feature importance
    if isinstance(mean_abs_shap, np.ndarray) and mean_abs_shap.ndim > 0:
        mean_abs_shap = mean_abs_shap.flatten()
    
    feature_importance = dict(zip(expanded_feature_names, mean_abs_shap))
    sorted_features = sorted(feature_importance.items(), key=lambda x: float(x[1]), reverse=True)
    
    # Save rankings to file
    with open(Path(results_dir) / 'siamese_feature_rankings.txt', 'w') as f:
        f.write("Siamese Feature Rankings (by importance):\n")
        f.write("=" * 50 + "\n")
        for feature, importance in sorted_features:
            f.write(f"{feature}: {float(importance):.6f}\n")
    
    return feature_importance

def compare_feature_importance(cnn_importance, siamese_importance, results_dir):
    """Compare feature importance between CNN and Siamese models."""
    logger.info("Comparing feature importance between models...")
    
    # Get common features
    features = set(cnn_importance.keys()) & set(siamese_importance.keys())
    
    # Calculate correlation between importance scores
    cnn_scores = [cnn_importance[f] for f in features]
    siamese_scores = [siamese_importance[f] for f in features]
    correlation = np.corrcoef(cnn_scores, siamese_scores)[0, 1]
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(cnn_scores, siamese_scores, alpha=0.5)
    plt.xlabel('CNN Feature Importance')
    plt.ylabel('Siamese Feature Importance')
    plt.title(f'Feature Importance Comparison (Correlation: {correlation:.3f})')
    
    # Add feature labels
    for i, feature in enumerate(features):
        plt.annotate(feature, (cnn_scores[i], siamese_scores[i]))
    
    plt.tight_layout()
    
    # Save plot
    comparison_file = Path(results_dir) / 'feature_importance_comparison.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance comparison plot saved to {comparison_file}")
    
    # Save comparison results
    with open(Path(results_dir) / 'feature_importance_comparison.txt', 'w') as f:
        f.write("Feature Importance Comparison:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Correlation between models: {correlation:.3f}\n\n")
        f.write("Feature\t\tCNN Importance\tSiamese Importance\n")
        f.write("-" * 50 + "\n")
        for feature in sorted(features, key=lambda x: cnn_importance[x], reverse=True):
            f.write(f"{feature}\t\t{cnn_importance[feature]:.6f}\t{siamese_importance[feature]:.6f}\n")
    
    return correlation 