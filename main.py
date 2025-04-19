import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import uvicorn
import yaml
from pathlib import Path

def main():
    """Run the FastAPI server."""
    # Load configuration
    config_path = Path('config/config.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required directories exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Run FastAPI server
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

if __name__ == "__main__":
    main()
