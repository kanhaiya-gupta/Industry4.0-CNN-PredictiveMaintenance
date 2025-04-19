# FastAPI application entry point

from src.api.app import app
from src.config_loader import ConfigLoader
import uvicorn

if __name__ == "__main__":
    config = ConfigLoader()
    api_config = config.get_api_config()
    uvicorn.run(app, host=api_config['host'], port=api_config['port'])
