from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class CaseBase(BaseModel):
    patient_clinical_id: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    anatomic_site: Optional[str] = None
    notes: Optional[str] = None


class CaseCreate(CaseBase):
    pass


class CaseUpdate(CaseBase):
    pass


class CaseResponse(CaseBase):
    id: str
    user_id: str
    created_at: datetime
    image_count: Optional[int] = 0

    class Config:
        from_attributes = True
