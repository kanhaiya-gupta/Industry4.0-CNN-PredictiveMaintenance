from fastapi import FastAPI
from src.config_loader import ConfigLoader
from .routes import health_router, predict_router

# Load configuration
config = ConfigLoader()

app = FastAPI(
    title=config.get_api_config()['title'],
    description=config.get_api_config()['description'],
    version=config.get_api_config()['version']
)

# Include routers
app.include_router(health_router)
app.include_router(predict_router) 