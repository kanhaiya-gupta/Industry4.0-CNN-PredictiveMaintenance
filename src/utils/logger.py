import logging
import sys
from pathlib import Path

def setup_logging(name='predictive_maintenance'):
    """Setup logging configuration to prevent duplicate logs"""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger(name)
    
    # Only add handlers if they haven't been added before
    if not logger.handlers:
        # Set the logging level
        logger.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Create and configure file handler
        file_handler = logging.FileHandler(
            logs_dir / f'{name}_{logging.getLevelName(logging.INFO)}.log'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger 