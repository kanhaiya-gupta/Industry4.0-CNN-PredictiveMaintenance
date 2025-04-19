"""
Prediction endpoint for the API.
"""

from fastapi import APIRouter, UploadFile, File
from ..services.predictor import PredictorService

router = APIRouter()
predictor = PredictorService()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make predictions on uploaded sensor data."""
    try:
        contents = await file.read()
        predictions = predictor.predict(contents)
        return predictions
    except Exception as e:
        return {"error": str(e)}
