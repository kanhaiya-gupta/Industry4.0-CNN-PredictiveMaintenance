# Industry 4.0 Predictive Maintenance using CNN and Siamese Networks

This project implements a predictive maintenance system using Convolutional Neural Networks (CNN) and Siamese Networks to detect anomalies in industrial equipment. The system analyzes sensor data to predict potential failures and maintenance needs, helping to prevent unplanned downtime and optimize maintenance schedules.

## Features

- **Dual Model Architecture**:
  - CNN for direct anomaly detection
  - Siamese Network for similarity-based anomaly detection
- **Comprehensive Data Processing**:
  - Feature extraction and normalization
  - Time series data handling
  - Train/test split with proper validation
- **Advanced Visualization**:
  - Training history plots
  - Confusion matrices
  - Model comparison metrics
  - t-SNE embeddings visualization
- **Model Evaluation**:
  - Accuracy metrics
  - ROC curves
  - Precision-Recall analysis
  - Feature importance visualization

## Project Structure

```
Industry4.0-CNN-PredictiveMaintenance/
├── config/
│   └── config.yaml          # Configuration parameters
├── data/
│   ├── raw/                 # Raw sensor data
│   └── processed/           # Processed datasets
├── notebooks/
│   └── PM_Siamese_Project.ipynb  # Jupyter notebook for analysis
├── results/                 # Outputs and visualizations
├── src/
│   ├── data_processor.py    # Data preprocessing and loading
│   ├── evaluation.py        # Model evaluation metrics
│   ├── siamese_model.py     # Siamese network implementation
│   ├── train.py            # Training pipeline
│   └── visualization.py     # Visualization utilities
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- t-SNE

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config/config.yaml`) to manage parameters:

```yaml
data:
  input_shape: [100, 1]  # Input data shape
  features: [feature1, feature2, ...]  # List of features to use
  test_size: 0.2  # Test set size

model:
  cnn:
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
  siamese:
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    num_pairs: 1000  # Number of pairs for training
```

## Usage

1. **Data Preparation**:
   - Place your raw sensor data in the `data/raw/` directory
   - The data processor will handle normalization and feature extraction

2. **Training**:
   ```bash
   python src/train.py
   ```
   This will:
   - Train both CNN and Siamese models
   - Generate evaluation metrics
   - Create visualizations in the `results/` directory

3. **Evaluation**:
   - View generated plots in the `results/` directory
   - Check model performance metrics in the console output
   - Compare CNN and Siamese model performance

## Results

The training process generates several outputs:
- Training history plots (loss and accuracy)
- Confusion matrices for both models
- Model comparison visualization
- t-SNE embeddings of the learned features
- ROC curves and precision-recall plots

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Industry 4.0 predictive maintenance research
- Inspired by Siamese network architectures for anomaly detection
- Uses standard machine learning practices for industrial applications
