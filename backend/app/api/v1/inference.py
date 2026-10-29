from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.models.case import Case
from backend.app.models.image import ImageRecord
from backend.app.models.prediction import PredictionResult
from backend.app.schemas.prediction_schema import PredictionRequest, PredictionResponse
from backend.app.core.dependencies import get_current_user
from backend.app.services.model_service import ModelService
from ml.models import CLASS_DESCRIPTIONS

router = APIRouter(tags=["Inference & Classification"])


@router.post("/images/{image_id}/predict", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
async def predict_image(
    image_id: str,
    pred_req: PredictionRequest = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute deep learning skin disease classification pipeline.
    Preprocesses and passes the lesion image through the trained CNN classifier
    (ResNet50 Baseline or EfficientNet-B0), producing calibrated class probabilities,
    predicted diagnostic label, and clinical risk category (LOW / MODERATE / HIGH / CRITICAL).
    """
    model_name = pred_req.model_name if pred_req and pred_req.model_name else "resnet50_baseline"

    # Fetch image and parent case
    result = await db.execute(
        select(ImageRecord)
        .options(selectinload(ImageRecord.case), selectinload(ImageRecord.preprocessing_log))
        .where(ImageRecord.id == image_id)
    )
    image_record = result.scalar_one_or_none()

    if not image_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{image_id}' not found."
        )

    # Enforce case-level authorization
    if current_user.role != "admin" and image_record.case.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to run inference on this case."
        )

    # Use preprocessed 224x224 image if available, else original
    target_path = image_record.file_path
    if image_record.preprocessing_log and image_record.preprocessing_log.processed_file_path:
        target_path = image_record.preprocessing_log.processed_file_path

    # Execute deep learning inference
    inference_result = ModelService.predict(target_path, model_name=model_name)

    # Record in database
    prediction = PredictionResult(
        image_id=image_record.id,
        model_name=inference_result["model_name"],
        model_version=inference_result["model_version"],
        predicted_class=inference_result["predicted_class"],
        confidence_score=inference_result["confidence_score"],
        class_probabilities=inference_result["class_probabilities"],
        risk_category=inference_result["risk_category"],
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)

    return PredictionResponse(
        id=prediction.id,
        image_id=prediction.image_id,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        predicted_class=prediction.predicted_class,
        predicted_description=CLASS_DESCRIPTIONS.get(prediction.predicted_class, prediction.predicted_class),
        confidence_score=prediction.confidence_score,
        class_probabilities=prediction.class_probabilities,
        risk_category=prediction.risk_category,
        created_at=prediction.created_at,
    )


@router.get("/images/{image_id}/predictions", response_model=List[PredictionResponse])
async def get_image_predictions(
    image_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve all historical predictions recorded for a given image."""
    result = await db.execute(
        select(ImageRecord).options(selectinload(ImageRecord.case)).where(ImageRecord.id == image_id)
    )
    image_record = result.scalar_one_or_none()

    if not image_record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    if current_user.role != "admin" and image_record.case.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    pred_res = await db.execute(
        select(PredictionResult)
        .where(PredictionResult.image_id == image_id)
        .order_by(PredictionResult.created_at.desc())
    )
    preds = pred_res.scalars().all()

    return [
        PredictionResponse(
            id=p.id,
            image_id=p.image_id,
            model_name=p.model_name,
            model_version=p.model_version,
            predicted_class=p.predicted_class,
            predicted_description=CLASS_DESCRIPTIONS.get(p.predicted_class, p.predicted_class),
            confidence_score=p.confidence_score,
            class_probabilities=p.class_probabilities,
            risk_category=p.risk_category,
            created_at=p.created_at,
        )
        for p in preds
    ]
