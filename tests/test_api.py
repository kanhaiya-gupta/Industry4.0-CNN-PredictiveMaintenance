import pytest
from fastapi.testclient import TestClient
from src.api.app import app
import torch
import numpy as np

client = TestClient(app)

@pytest.fixture
def sample_sensor_data():
    """Create sample sensor data for testing."""
    return {
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

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Predictive Maintenance API"}

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "cnn_model_loaded" in data
    assert "siamese_model_loaded" in data

def test_predict_endpoint(sample_sensor_data):
    """Test the predict endpoint."""
    response = client.post("/predict", json=sample_sensor_data)
    assert response.status_code == 200
    data = response.json()
    assert "cnn_prediction" in data
    assert "siamese_prediction" in data
    assert "confidence" in data
    assert isinstance(data["cnn_prediction"], int)
    assert isinstance(data["siamese_prediction"], int)
    assert isinstance(data["confidence"], float)

def test_predict_endpoint_invalid_data():
    """Test the predict endpoint with invalid data."""
    invalid_data = {"invalid": "data"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_train_endpoint():
    """Test the train endpoint."""
    response = client.post("/train")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "Training completed successfully" in data["message"]

def test_predict_endpoint_missing_models():
    """Test the predict endpoint when models are not loaded."""
    # Temporarily set models to None
    app.state.cnn_model = None
    app.state.siamese_model = None
    
    response = client.post("/predict", json=sample_sensor_data)
    assert response.status_code == 503  # Service Unavailable
    assert "Models not loaded" in response.json()["detail"]
    
    # Restore models
    app.state.cnn_model = torch.nn.Module()
    app.state.siamese_model = torch.nn.Module() 