import pytest
import torch
import numpy as np
from src.siamese_model import SiameseNetwork

@pytest.fixture
def sample_pairs():
    """Create sample pairs for testing."""
    return torch.randn(32, 2, 19, 1)  # batch_size=32, pairs=2, features=19, sequence_length=1

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return torch.randint(0, 2, (32,))  # binary labels for pairs

def test_siamese_model_initialization():
    """Test Siamese model initialization."""
    model = SiameseNetwork(input_shape=(19, 1))
    assert model.input_shape == (19, 1)
    assert isinstance(model.model, torch.nn.Module)

def test_forward_pass(sample_pairs):
    """Test model forward pass."""
    model = SiameseNetwork(input_shape=(19, 1))
    output = model.model(sample_pairs)
    assert output.shape == (32, 1)  # batch_size=32, similarity score

def test_predict(sample_pairs):
    """Test model prediction."""
    model = SiameseNetwork(input_shape=(19, 1))
    predictions = model.predict(sample_pairs.numpy())
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (32,)

def test_save_and_load(tmp_path):
    """Test model saving and loading."""
    # Create and save model
    model = SiameseNetwork(input_shape=(19, 1))
    save_path = tmp_path / "test_siamese_model.pth"
    model.save_model(save_path)
    
    # Load model
    loaded_model = SiameseNetwork(input_shape=(19, 1))
    loaded_model.load_model(save_path)
    
    # Compare model parameters
    for p1, p2 in zip(model.model.parameters(), loaded_model.model.parameters()):
        assert torch.allclose(p1, p2)

def test_training_step(sample_pairs, sample_labels):
    """Test model training step."""
    model = SiameseNetwork(input_shape=(19, 1))
    model.compile_model()
    
    # Create a simple training loop
    optimizer = torch.optim.Adam(model.model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Forward pass
    outputs = model.model(sample_pairs)
    loss = criterion(outputs, sample_labels.float().unsqueeze(1))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)

def test_embedding_shape():
    """Test embedding shape."""
    model = SiameseNetwork(input_shape=(19, 1))
    sample_input = torch.randn(1, 19, 1)
    embedding = model.get_embedding(sample_input)
    assert embedding.shape == (1, 128)  # embedding_size=128 