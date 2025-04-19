"""
Health check endpoint for the API.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {"status": "healthy"} 