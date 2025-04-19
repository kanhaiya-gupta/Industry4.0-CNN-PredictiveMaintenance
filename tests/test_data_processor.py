import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data_processor import DataProcessor

@pytest.fixture
def sample_data():
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

def test_data_processor_initialization():
    """Test DataProcessor initialization."""
    data_path = Path('data/raw/HRSS_normal_standard.csv')
    processor = DataProcessor(data_path)
    assert processor.data_path == data_path

def test_load_data(sample_data, tmp_path):
    """Test data loading functionality."""
    # Save sample data to temporary file
    test_file = tmp_path / "test_data.csv"
    sample_data.to_csv(test_file, index=False)
    
    processor = DataProcessor(test_file)
    loaded_data = processor.load_data()
    
    assert isinstance(loaded_data, tuple)
    assert len(loaded_data) == 2
    assert isinstance(loaded_data[0], np.ndarray)
    assert isinstance(loaded_data[1], np.ndarray)

def test_preprocess_data(sample_data):
    """Test data preprocessing."""
    processor = DataProcessor(None)  # We'll use the sample data directly
    X, y = processor.preprocess_data((sample_data.drop('Labels', axis=1).values, 
                                    sample_data['Labels'].values))
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[1] == 19  # Number of features
    assert len(y) == len(sample_data)

def test_create_siamese_pairs(sample_data):
    """Test Siamese pairs creation."""
    processor = DataProcessor(None)
    pairs, labels = processor.create_siamese_pairs(
        sample_data.drop('Labels', axis=1).values,
        sample_data['Labels'].values
    )
    
    assert isinstance(pairs, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert pairs.shape[0] == labels.shape[0]
    assert pairs.shape[1] == 2  # Each pair has two samples 