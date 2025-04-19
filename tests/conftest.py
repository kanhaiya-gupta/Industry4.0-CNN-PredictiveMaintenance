import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import os
import yaml
from src.utils.logger import setup_logging

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def test_models_dir(tmp_path_factory):
    """Create a temporary directory for test models."""
    return tmp_path_factory.mktemp("test_models")

@pytest.fixture(scope="session")
def sample_sensor_data():
    """Create sample sensor data for testing."""
    return pd.DataFrame({
        'Timestamp': [0.0, 0.1, 0.2],
        'Labels': [0, 1, 0],
        'I_w_BLO_Weg': [-107, -108, -106],
        'O_w_BLO_power': [0, 0, 0],
        'O_w_BLO_voltage': [0, 0, 0],
        'I_w_BHL_Weg': [0, 0, 0],
        'O_w_BHL_power': [0, 0, 0],
        'O_w_BHL_voltage': [0, 0, 0],
        'I_w_BHR_Weg': [-1268, -1269, -1267],
        'O_w_BHR_power': [0, 0, 0],
        'O_w_BHR_voltage': [0, 0, 0],
        'I_w_BRU_Weg': [-26, -27, -25],
        'O_w_BRU_power': [84, 85, 83],
        'O_w_BRU_voltage': [11, 11, 11],
        'I_w_HR_Weg': [0, 0, 0],
        'O_w_HR_power': [7168, 7169, 7167],
        'O_w_HR_voltage': [26, 26, 26],
        'I_w_HL_Weg': [0, 0, 0],
        'O_w_HL_power': [7720, 7721, 7719],
        'O_w_HL_voltage': [24, 24, 24]
    })

@pytest.fixture(scope="session")
def sample_input_tensor():
    """Create sample input tensor for testing."""
    return torch.randn(32, 19, 1)  # batch_size=32, features=19, sequence_length=1

@pytest.fixture(scope="session")
def sample_labels_tensor():
    """Create sample labels tensor for testing."""
    return torch.randint(0, 4, (32,))  # 4 classes

@pytest.fixture(scope="session")
def sample_siamese_pairs():
    """Create sample pairs for Siamese network testing."""
    return torch.randn(32, 2, 19, 1)  # batch_size=32, pairs=2, features=19, sequence_length=1

@pytest.fixture(scope="session")
def sample_siamese_labels():
    """Create sample labels for Siamese network testing."""
    return torch.randint(0, 2, (32,))  # binary labels for pairs

@pytest.fixture(scope="session")
def config():
    """Load configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

@pytest.fixture(scope="session")
def logger():
    """Setup logger for tests"""
    return setup_logging("tests")

@pytest.fixture(scope="session")
def models_dir(config):
    """Ensure models directory exists"""
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

@pytest.fixture(scope="session")
def data_dir(config):
    """Ensure data directory exists"""
    data_dir = Path(config["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir 