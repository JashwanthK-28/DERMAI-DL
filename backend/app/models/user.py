import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
from backend.app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="clinician")  # patient, clinician, researcher, admin
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    cases = relationship("Case", back_populates="user", cascade="all, delete-orphan")
