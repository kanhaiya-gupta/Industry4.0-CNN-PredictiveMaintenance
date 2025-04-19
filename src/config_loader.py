import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {str(e)}")

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        return self.config['model'][model_type]

    def get_data_config(self, data_type: str) -> Dict[str, Any]:
        """Get configuration for a specific data type."""
        return self.config['data'][data_type]

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config['evaluation']

    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config['visualization']

    def get_path_config(self) -> Dict[str, Any]:
        """Get path configuration."""
        return self.config['paths']

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config['api']

    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config 