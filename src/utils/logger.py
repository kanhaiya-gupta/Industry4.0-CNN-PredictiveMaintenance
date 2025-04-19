import logging
from datetime import datetime
from pathlib import Path

def setup_logging(module_name: str = 'predictive_maintenance'):
    """
    Set up logging configuration for the application.
    
    Args:
        module_name (str): Name of the module using the logger (e.g., 'training', 'api')
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('predictive_maintenance')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f'{module_name}_{current_time}.log',
        encoding='utf-8'
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 