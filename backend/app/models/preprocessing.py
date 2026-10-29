import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database import Base


class PreprocessingLog(Base):
    __tablename__ = "preprocessing_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("image_records.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    processed_storage_key = Column(String(255), nullable=False)
    processed_file_path = Column(String(512), nullable=False)
    blur_score = Column(Float, nullable=False)
    brightness_score = Column(Float, nullable=False)
    contrast_score = Column(Float, nullable=False)
    quality_status = Column(String(20), nullable=False)  # PASS, WARNING, REJECT
    quality_reasons = Column(JSON, nullable=False, default=list)  # list of string reasons
    original_dim = Column(String(50), nullable=False)  # e.g. "1024x768"
    processed_dim = Column(String(50), nullable=False, default="224x224")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    image = relationship("ImageRecord", back_populates="preprocessing_log")
