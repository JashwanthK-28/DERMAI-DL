from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.models.case import Case
from backend.app.models.image import ImageRecord
from backend.app.schemas.case_schema import CaseCreate, CaseUpdate, CaseResponse
from backend.app.core.dependencies import get_current_user, get_case_for_user

router = APIRouter(prefix="/cases", tags=["Case Management"])


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    case_in: CaseCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new clinical skin lesion case for the current user."""
    case = Case(
        user_id=current_user.id,
        patient_clinical_id=case_in.patient_clinical_id,
        age=case_in.age,
        sex=case_in.sex,
        anatomic_site=case_in.anatomic_site,
        notes=case_in.notes
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)
    return case


@router.get("", response_model=List[CaseResponse])
async def list_cases(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List clinical cases.
    Non-admin users can ONLY list cases where case.user_id == current_user.id.
    """
    query = select(Case)
    if current_user.role != "admin":
        query = query.where(Case.user_id == current_user.id)
    query = query.order_by(Case.created_at.desc())

    result = await db.execute(query)
    cases = result.scalars().all()
    return cases


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(case: Case = Depends(get_case_for_user)):
    """
    Retrieve case details.
    Strictly enforces 403 Forbidden if non-admin user attempts to access another user's case.
    """
    return case


@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_update: CaseUpdate,
    case: Case = Depends(get_case_for_user),
    db: AsyncSession = Depends(get_db)
):
    """Update case details (authorized owner or admin only)."""
    update_data = case_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(case, field, value)

    await db.commit()
    await db.refresh(case)
    return case


@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case(
    case: Case = Depends(get_case_for_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a case and all associated images (authorized owner or admin only)."""
    await db.delete(case)
    await db.commit()
    return None
