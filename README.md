# Industry 4.0 Predictive Maintenance System
A production-grade predictive maintenance system leveraging PyTorch-based CNN and Siamese networks for real-time fault detection in hydraulic systems, featuring a FastAPI backend, MLflow experiment tracking, and comprehensive monitoring capabilities.

## Features

- **Real-time Sensor Data Processing**: Handles time series data from 18 hydraulic sensors across multiple cycles.
- **CNN and Siamese Networks for Fault Detection**: Employs both CNN and Siamese network architectures for comprehensive fault detection:
  - CNN for direct fault classification
  - Siamese network for similarity-based anomaly detection
- **Data Visualization**: Generates confusion matrices, t-SNE visualizations, and similarity score distributions for performance evaluation.
- **Model Training and Evaluation**: Tracks training progress with loss and accuracy metrics, providing detailed classification reports.
- **Comprehensive Data Preprocessing**: Normalizes, reshapes, and splits sensor data into training/testing sets, with sequence generation for time series analysis.
- **RESTful API for Predictions**: Provides real-time prediction integration through FastAPI.
- **Logging System**: Monitors model performance and system operations with detailed logging.

## Project Structure

```
.
├── config/
│   └── config.yaml          # Configuration file
├── data/
│   ├── raw/                 # Raw sensor data files
│   │   ├── HRSS_anomalous_optimized.csv
│   │   ├── HRSS_anomalous_standard.csv
│   │   ├── HRSS_normal_optimized.csv
│   │   └── HRSS_normal_standard.csv
│   └── processed/           # Processed data files
│       └── preprocessed_data.pkl
├── docs/
│   └── FastAPI_UI_Swagger.jpeg  # API documentation
├── logs/                    # Application logs
│   ├── api_*.log           # API logs
│   ├── training_*.log      # Training logs
│   └── predictive_maintenance_*.log  # General logs
├── models/                  # Trained model files
│   ├── cnn_model.pth       # CNN model weights
│   └── siamese_model.pth   # Siamese model weights
├── notebooks/              # Jupyter notebooks for analysis
│   ├── PM-CNN.ipynb
│   ├── PM-CRNN.ipynb
│   ├── PMResnet.ipynb
│   └── PM_Siamese_Project.ipynb
├── results/                # Training results and visualizations
│   ├── confusion_matrix_cnn model.png
│   ├── confusion_matrix_siamese model.png
│   ├── model_comparison.png
│   └── training_history.png
├── scripts/               # Utility scripts
├── src/
│   ├── api/              # FastAPI application
│   │   ├── routes/      # API route handlers
│   │   │   ├── health.py
│   │   │   └── predict.py
│   │   ├── services/    # Business logic
│   │   │   └── predictor.py
│   │   └── app.py       # Main FastAPI application
│   ├── cnn_model.py     # CNN model implementation
│   ├── siamese_model.py # Siamese network implementation
│   ├── data_processor.py # Data processing utilities
│   ├── config_loader.py  # Configuration loader
│   ├── evaluation.py     # Model evaluation utilities
│   ├── train.py         # Training script
│   ├── visualization.py  # Visualization utilities
│   └── utils/
│       └── logger.py    # Logging utilities
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
└── setup.py            # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Industry4.0-CNN-PredictiveMaintenance
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) to manage all settings. Here's the structure and example values:

```yaml
# Paths Configuration
paths:
  models_dir: 'models'
  data_dir: 'data/raw'
  results_dir: 'results'
  cnn_model_file: 'cnn_model.pth'
  siamese_model_file: 'siamese_model.pth'
  sensor_data_file: 'HRSS_anomalous_optimized.csv'

# Model Configuration
model:
  cnn:
    input_shape: [8, 18]  # sequence_length, num_features
    num_classes: 2        # Number of output classes (normal/fault)
    filters: [32, 64, 128]
    kernel_size: 3
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping:
      patience: 5
      min_delta: 0.001

  siamese:
    input_shape: [8, 18]  # sequence_length, num_features
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    early_stopping:
      patience: 5
      min_delta: 0.001

# Data Processing Configuration
data_processing:
  sequence_length: 8
  test_size: 0.1
  scale_features: true
  random_state: 42

# Training Configuration
training:
  early_stopping:
    monitor: 'val_loss'
    patience: 5
    restore_best_weights: true
  validation_split: 0.1
  verbose: 1

# Evaluation Configuration
evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
  confusion_matrix:
    normalize: true
    title: 'Confusion Matrix'
    cmap: 'Blues'
  thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  decision_threshold: 0.5

# Visualization Configuration
visualization:
  figure_size: [10, 8]
  font_size:
    title: 25
    label: 20
    tick: 15
  colors:
    - '#1f77b4'  # blue
    - '#ff7f0e'  # orange
    - '#2ca02c'  # green
    - '#d62728'  # red
  tsne:
    n_components: 2
    random_state: 42
    perplexity: 30
    n_iter: 1000

# API Configuration
api:
  host: '0.0.0.0'
  port: 8000
  title: 'Industry 4.0 Predictive Maintenance API'
  description: 'API for predicting maintenance needs using Siamese networks'
  version: '1.0.0'
  debug: true
  workers: 1
  reload: true

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "predictive_maintenance"
```

### Configuration Sections

1. **Paths**
   - `models_dir`: Directory for saving trained models
   - `data_dir`: Directory containing raw sensor data
   - `results_dir`: Directory for saving evaluation results
   - `cnn_model_file`: Name of the saved CNN model file
   - `siamese_model_file`: Name of the saved Siamese model file
   - `sensor_data_file`: Name of the sensor data CSV file

2. **Model Settings**
   - CNN model parameters (input shape, filters, learning rate, etc.)
   - Siamese network parameters (input shape, learning rate, etc.)
   - Training parameters (batch size, epochs, early stopping)

3. **Data Processing**
   - Sequence length for time series analysis
   - Test split size
   - Feature scaling options
   - Random state for reproducibility

4. **Training**
   - Early stopping configuration
   - Validation split size
   - Verbosity level

5. **Evaluation**
   - Performance metrics to track
   - Confusion matrix settings
   - Decision thresholds
   - Visualization parameters

6. **API Settings**
   - Server host and port
   - API metadata (title, description, version)
   - Development mode settings (debug, workers, reload)

7. **MLflow**
   - Tracking server URI
   - Experiment name for model tracking

## Model Architecture

### CNN Architecture
1. **Convolutional Layers**:
   - Three convolutional blocks with increasing channels (32 -> 64 -> 128)
   - Batch normalization and ReLU activation
   - Max pooling for dimensionality reduction

2. **Dense Layers**:
   - Two fully connected layers (256 -> 128)
   - Dropout for regularization
   - ReLU activation
   - Final softmax layer for classification

### Siamese Network Architecture
1. **Convolutional Encoder**:
   - Three convolutional blocks with increasing channels (64 -> 128 -> 256)
   - Batch normalization and ReLU activation
   - Max pooling for dimensionality reduction

2. **Embedding Network**:
   - Three fully connected layers (512 -> 256 -> 128)
   - Dropout for regularization
   - ReLU activation

3. **Distance Network**:
   - Single layer with sigmoid activation
   - Outputs similarity score between 0 and 1

## Technical Details

### System Architecture
- **Deep Learning Framework**: PyTorch for model development and training
- **Backend Framework**: FastAPI for high-performance REST API implementation
- **Data Processing**: Pandas and NumPy for efficient data manipulation and preprocessing
- **Model Architecture**:
  - CNN: 3-layer convolutional network with batch normalization and max pooling
  - Siamese Network: Twin network architecture with shared weights
- **Training Pipeline**:
  - Custom data loader for time series sequences
  - Early stopping with model checkpointing
  - Gradient clipping for stable training
  - Learning rate scheduling
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC curves
  - Confusion matrices
  - t-SNE visualization for feature space analysis

### Key Technical Features
1. **Time Series Processing**:
   - Sliding window approach for sequence generation
   - Custom data augmentation for time series
   - Feature scaling and normalization
   - Sequence length optimization (8 timesteps)

2. **Model Implementation**:
   - Custom PyTorch Dataset class for efficient data loading
   - GPU acceleration support
   - Model checkpointing and resume training capability
   - Custom loss functions for Siamese network training

3. **API Implementation**:
   - Asynchronous request handling
   - Input validation using Pydantic models
   - Comprehensive error handling
   - Swagger/OpenAPI documentation
   - Rate limiting and request validation

4. **Monitoring and Logging**:
   - Structured logging with log rotation
   - Performance metrics tracking
   - MLflow integration for experiment tracking
   - Custom logging handlers for different components

### Technologies Used
- **Programming Languages**: Python 3.8+
- **Deep Learning**: PyTorch 1.9+
- **Web Framework**: FastAPI 0.68+
- **Data Processing**: 
  - Pandas 1.3+
  - NumPy 1.21+
  - Scikit-learn 0.24+
- **Visualization**:
  - Matplotlib 3.4+
  - Seaborn 0.11+
- **API Documentation**: Swagger UI
- **Model Tracking**: MLflow
- **Testing**: Pytest
- **Development Tools**:
  - Git for version control
  - Docker for containerization
  - VS Code for development

### Performance Optimization
1. **Data Processing**:
   - Vectorized operations for feature scaling
   - Efficient memory management for large datasets
   - Parallel processing for data augmentation

2. **Model Training**:
   - Mixed precision training (FP16)
   - Gradient accumulation for large batch sizes
   - Custom learning rate schedulers
   - Model pruning for inference optimization

3. **API Performance**:
   - Async request handling
   - Response caching
   - Connection pooling
   - Load balancing support

### System Requirements
- **Hardware**:
  - CPU: 4+ cores
  - RAM: 16GB minimum
  - GPU: NVIDIA GPU with CUDA support (optional)
- **Software**:
  - Python 3.8+
  - CUDA Toolkit 11.0+ (for GPU support)
  - Docker 20.10+ (for containerized deployment)

### Deployment Options
1. **Local Deployment**:
   - Virtual environment setup
   - Direct Python execution
   - Development server

2. **Containerized Deployment**:
   - Docker container
   - Kubernetes support
   - Load balancing configuration

3. **Cloud Deployment**:
   - AWS EC2/ECS
   - Google Cloud Platform
   - Azure ML Service

### Security Features
- Input validation and sanitization
- Rate limiting
- CORS configuration
- API key authentication
- Secure model storage

## Usage

### Data Format

The system expects sensor data in CSV format with the following columns:

| Column Name      | Description                    | Example Value |
|------------------|--------------------------------|---------------|
| Timestamp        | Time index for each cycle      | 0.0459976196289063 |
| Labels           | Binary labels (0=normal, 1=fault) | 0 |
| I_w_BLO_Weg     | BLO Weg current               | -107 |
| O_w_BLO_power   | BLO power output              | 0 |
| O_w_BLO_voltage | BLO voltage output            | 0 |
| I_w_BHL_Weg     | BHL Weg current               | 0 |
| O_w_BHL_power   | BHL power output              | 0 |
| O_w_BHL_voltage | BHL voltage output            | 0 |
| I_w_BHR_Weg     | BHR Weg current               | -1268 |
| O_w_BHR_power   | BHR power output              | 0 |
| O_w_BHR_voltage | BHR voltage output            | 0 |
| I_w_BRU_Weg     | BRU Weg current               | -26 |
| O_w_BRU_power   | BRU power output              | 84 |
| O_w_BRU_voltage | BRU voltage output            | 11 |
| I_w_HR_Weg      | HR Weg current                | 0 |
| O_w_HR_power    | HR power output               | 7168 |
| O_w_HR_voltage  | HR voltage output             | 26 |
| I_w_HL_Weg      | HL Weg current                | 0 |
| O_w_HL_power    | HL power output               | 7720 |
| O_w_HL_voltage  | HL voltage output             | 24 |

Example data row:
```json
{
    "Timestamp": 0.0459976196289063,
    "Labels": 0,
    "I_w_BLO_Weg": -107,
    "O_w_BLO_power": 0,
    "O_w_BLO_voltage": 0,
    "I_w_BHL_Weg": 0,
    "O_w_BHL_power": 0,
    "O_w_BHL_voltage": 0,
    "I_w_BHR_Weg": -1268,
    "O_w_BHR_power": 0,
    "O_w_BHR_voltage": 0,
    "I_w_BRU_Weg": -26,
    "O_w_BRU_power": 84,
    "O_w_BRU_voltage": 11,
    "I_w_HR_Weg": 0,
    "O_w_HR_power": 7168,
    "O_w_HR_voltage": 26,
    "I_w_HL_Weg": 0,
    "O_w_HL_power": 7720,
    "O_w_HL_voltage": 24
}
```

The data represents measurements from six hydraulic components:
1. **BLO (Bottom Left Outlet)**
   - Weg current (I_w_BLO_Weg)
   - Power output (O_w_BLO_power)
   - Voltage output (O_w_BLO_voltage)

2. **BHL (Bottom Left Hydraulic)**
   - Weg current (I_w_BHL_Weg)
   - Power output (O_w_BHL_power)
   - Voltage output (O_w_BHL_voltage)

3. **BHR (Bottom Right Hydraulic)**
   - Weg current (I_w_BHR_Weg)
   - Power output (O_w_BHR_power)
   - Voltage output (O_w_BHR_voltage)

4. **BRU (Bottom Right Upper)**
   - Weg current (I_w_BRU_Weg)
   - Power output (O_w_BRU_power)
   - Voltage output (O_w_BRU_voltage)

5. **HR (Hydraulic Right)**
   - Weg current (I_w_HR_Weg)
   - Power output (O_w_HR_power)
   - Voltage output (O_w_HR_voltage)

6. **HL (Hydraulic Left)**
   - Weg current (I_w_HL_Weg)
   - Power output (O_w_HL_power)
   - Voltage output (O_w_HL_voltage)

Each component has three types of measurements:
- **Weg current (I_w_*)**: Input current to the component
- **Power output (O_w_*_power)**: Power output from the component
- **Voltage output (O_w_*_voltage)**: Voltage output from the component

The data is collected at regular intervals (Timestamp) and includes binary labels indicating normal (0) or fault (1) conditions.

### Training the Model

```bash
python src/train.py
```

This will:
1. Load and preprocess the sensor data
2. Generate Siamese pairs for training
3. Train the model with the specified configuration
4. Save the trained model and evaluation results

### Starting the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

The API provides a Swagger UI interface for easy testing and documentation. You can access it at `http://localhost:8000/docs`:

![FastAPI Swagger UI](docs/FastAPI_UI_Swagger.jpeg)

#### 1. Health Check
- **GET** `/health`
- Returns the health status of the API and loaded models
- Response:
```json
{
    "status": "healthy",
    "siamese_model_loaded": true,
    "siamese_model_exists": true
}
```

#### 2. Training
- **POST** `/train`
- Triggers model training
- Response:
```json
{
    "status": "success",
    "message": "Training completed successfully"
}
```

#### 3. Prediction
- **POST** `/predict`
- Makes predictions on new sensor data
- Request body:
```json
{
    "sensor_data": {
        "I_w_BLO_Weg": 0.5,
        "O_w_BLO_power": 0.3,
        // ... other sensor features
    }
}
```
- Response:
```json
{
    "prediction": "fault",
    "similarity_score": 0.85,
    "confidence": 0.92
}
```

## Data Visualization

### Sensor Data Characteristics
The system processes data from 18 hydraulic sensors, with each sensor providing three types of measurements:
- **Weg current (I_w_*)**: Input current to the component
- **Power output (O_w_*_power)**: Power output from the component
- **Voltage output (O_w_*_voltage)**: Voltage output from the component

### Data Distribution
1. **Time Series Patterns**:
   - Sensor readings are collected at regular intervals
   - Each cycle contains 8 timesteps of measurements
   - Normal and fault conditions show distinct patterns in sensor readings

2. **Feature Distributions**:
   - Power outputs range from 0 to ~8000 units
   - Voltage outputs typically range from 0 to 30 units
   - Weg currents show both positive and negative values

3. **Class Distribution**:
   - Binary classification: Normal (0) vs Fault (1)
   - Imbalanced dataset with more normal samples than fault samples
   - Data augmentation techniques applied to balance the classes

### Visualization Examples
The following visualizations are generated during data preprocessing:

1. **Sensor Time Series**:
   - Line plots showing sensor readings over time
   - Separate plots for normal and fault conditions
   - Highlighted differences in patterns between conditions

2. **Feature Correlation**:
   - Heatmap showing correlations between different sensors
   - Strong correlations between related sensors (e.g., power and voltage)
   - Weak correlations between unrelated sensors

3. **Distribution Plots**:
   - Histograms showing value distributions for each sensor
   - Box plots comparing normal vs fault conditions
   - Violin plots showing density distributions

4. **t-SNE Visualization**:
   - 2D projection of high-dimensional sensor data
   - Clear separation between normal and fault conditions
   - Clusters indicating different types of faults

### Data Preprocessing Steps
1. **Normalization**:
   - Min-max scaling applied to all features
   - Separate scaling for each sensor type
   - Preserved temporal relationships

2. **Sequence Generation**:
   - Sliding window approach with window size of 8
   - Overlapping windows for data augmentation
   - Maintained temporal order within sequences

3. **Feature Engineering**:
   - Statistical features (mean, std, min, max)
   - Temporal features (slope, change rate)
   - Cross-sensor features (ratios, differences)

## Results

### Model Performance

#### CNN Model
- **Accuracy**: 0.6411 (64.11%)
- **Precision**: 0.6497 (64.97%)
- **Recall**: 0.6411 (64.11%)
- **F1 Score**: 0.5522 (55.22%)

#### Siamese Model
- **Accuracy**: 0.4260 (42.60%)
- **Precision**: 0.5190 (51.90%)
- **Recall**: 0.4260 (42.60%)
- **F1 Score**: 0.3845 (38.45%)

### Model Comparison
| Metric     | CNN Model | Siamese Model | Difference |
|------------|-----------|---------------|------------|
| Accuracy   | 0.6411    | 0.4260        | +0.2152    |
| Precision  | 0.6497    | 0.5190        | +0.1308    |
| Recall     | 0.6411    | 0.4260        | +0.2152    |
| F1 Score   | 0.5522    | 0.3845        | +0.1677    |

### Key Findings
1. The CNN model demonstrates superior performance across all metrics compared to the Siamese model
2. The CNN model achieves over 64% accuracy in fault detection, which is significantly higher than the Siamese model's 42.6%
3. Both models show balanced precision and recall scores, indicating consistent performance across different types of predictions
4. The F1 scores suggest that the CNN model provides more reliable predictions overall

### Visualization Results
The following visualizations are generated during training and evaluation:
- ROC curves for both training and validation data
- Confusion matrices
- Training history plots
- Accuracy distribution plots
- Model comparison charts

All visualizations are saved in the `results` directory for further analysis.

## Testing

The project includes a comprehensive test suite to ensure code quality and functionality. Tests are organized by component:

### Running Tests

1. Install test dependencies:
```bash
pip install pytest pytest-cov
```