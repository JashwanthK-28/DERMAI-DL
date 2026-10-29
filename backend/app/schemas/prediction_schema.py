from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    model_name: Optional[str] = "resnet50_baseline"  # or "efficientnet_b0"


class PredictionResponse(BaseModel):
    id: str
    image_id: str
    model_name: str
    model_version: str
    predicted_class: str
    predicted_description: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    risk_category: str  # LOW, MODERATE, HIGH, CRITICAL
    created_at: datetime

    class Config:
        from_attributes = True


class PreprocessingLogResponse(BaseModel):
    id: str
    image_id: str
    processed_storage_key: str
    processed_file_path: str
    blur_score: float
    brightness_score: float
    contrast_score: float
    quality_status: str
    quality_reasons: List[str]
    original_dim: str
    processed_dim: str
    created_at: datetime

    class Config:
        from_attributes = True
