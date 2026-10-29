import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.database import Base


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey("image_records.id", ondelete="CASCADE"), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False, default="1.0.0")
    predicted_class = Column(String(50), nullable=False)  # BCC, SCC, MEL, ACK, NEV, SEK
    confidence_score = Column(Float, nullable=False)  # [0.0, 1.0]
    class_probabilities = Column(JSON, nullable=False)  # {"ACK": 0.05, "BCC": 0.85, ...}
    risk_category = Column(String(20), nullable=False)  # LOW, MODERATE, HIGH, CRITICAL
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    image = relationship("ImageRecord", back_populates="prediction_results")
