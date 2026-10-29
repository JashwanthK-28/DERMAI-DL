from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class QualityAssessmentSchema(BaseModel):
    blur_score: float
    brightness_score: float
    contrast_score: float
    quality_status: str  # PASS, WARNING, REJECT
    quality_reasons: List[str]
    original_dim: str
    processed_dim: str


class ImageRecordResponse(BaseModel):
    id: str
    case_id: str
    original_filename: str
    storage_key: str
    file_path: str
    mime_type: str
    file_size_bytes: int
    width: int
    height: int
    created_at: datetime
    quality: Optional[QualityAssessmentSchema] = None

    class Config:
        from_attributes = True


class ImageUploadResponse(BaseModel):
    image: ImageRecordResponse
    preprocessing: QualityAssessmentSchema
    message: str
