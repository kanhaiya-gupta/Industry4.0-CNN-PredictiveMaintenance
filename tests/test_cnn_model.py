import pytest
import torch
import numpy as np
from src.cnn_model import CNNModel

@pytest.fixture
def sample_input():
    """Create sample input data for testing."""
    return torch.randn(32, 19, 1)  # batch_size=32, features=19, sequence_length=1

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return torch.randint(0, 4, (32,))  # 4 classes

def test_cnn_model_initialization():
    """Test CNN model initialization."""
    model = CNNModel(input_shape=(19, 1), num_classes=4)
    assert model.input_shape == (19, 1)
    assert model.num_classes == 4
    assert isinstance(model.model, torch.nn.Module)

def test_forward_pass(sample_input):
    """Test model forward pass."""
    model = CNNModel(input_shape=(19, 1), num_classes=4)
    output = model.model(sample_input)
    assert output.shape == (32, 4)  # batch_size=32, num_classes=4

def test_predict(sample_input):
    """Test model prediction."""
    model = CNNModel(input_shape=(19, 1), num_classes=4)
    predictions = model.predict(sample_input.numpy())
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (32,)

def test_save_and_load(tmp_path):
    """Test model saving and loading."""
    # Create and save model
    model = CNNModel(input_shape=(19, 1), num_classes=4)
    save_path = tmp_path / "test_cnn_model.pth"
    model.save_model(save_path)
    
    # Load model
    loaded_model = CNNModel(input_shape=(19, 1), num_classes=4)
    loaded_model.load_model(save_path)
    
    # Compare model parameters
    for p1, p2 in zip(model.model.parameters(), loaded_model.model.parameters()):
        assert torch.allclose(p1, p2)

def test_training_step(sample_input, sample_labels):
    """Test model training step."""
    model = CNNModel(input_shape=(19, 1), num_classes=4)
    model.compile_model()
    
    # Create a simple training loop
    optimizer = torch.optim.Adam(model.model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model.model(sample_input)
    loss = criterion(outputs, sample_labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss) 