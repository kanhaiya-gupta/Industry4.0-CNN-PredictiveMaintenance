"""
API routes for Industry 4.0 Predictive Maintenance system.
"""

from .health import router as health_router
from .predict import router as predict_router

__all__ = ['health_router', 'predict_router']
