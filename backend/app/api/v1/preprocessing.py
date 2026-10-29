from typing import List
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.models.case import Case
from backend.app.models.image import ImageRecord
from backend.app.models.preprocessing import PreprocessingLog
from backend.app.schemas.image_schema import ImageUploadResponse, ImageRecordResponse, QualityAssessmentSchema
from backend.app.core.dependencies import get_current_user, get_case_for_user
from backend.app.services.preprocessing_service import PreprocessingService

router = APIRouter(tags=["Image Upload & Preprocessing"])


@router.post("/cases/{case_id}/images/upload", response_model=ImageUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_image(
    case_id: str,
    file: UploadFile = File(...),
    case: Case = Depends(get_case_for_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and preprocess a clinical skin lesion mobile image.
    Executes the 8-step preprocessing engine:
    1. Preserve raw untouched original
    2. Decode & validate bytes (reject empty/corrupt with 400 Bad Request)
    3. EXIF orientation correction
    4. Quality assessment (blur, brightness, contrast scoring)
    5. RGB standardization
    6. Bicubic 224x224 resizing
    7. Pixel normalization to [0.0, 1.0]
    8. Storage & logging
    """
    raw_bytes = await file.read()
    if not raw_bytes or len(raw_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded. Payload contains zero bytes."
        )

    # Execute preprocessing engine
    proc_result = PreprocessingService.process_image(raw_bytes, file.filename or "uploaded.png")

    # Create ImageRecord in DB
    image_record = ImageRecord(
        case_id=case.id,
        original_filename=file.filename or "uploaded.png",
        storage_key=proc_result["original_storage_key"],
        file_path=proc_result["original_file_path"],
        mime_type=file.content_type or "image/png",
        file_size_bytes=len(raw_bytes),
        width=proc_result["original_width"],
        height=proc_result["original_height"],
    )
    db.add(image_record)
    await db.flush()

    # Create PreprocessingLog in DB
    prep_log = PreprocessingLog(
        image_id=image_record.id,
        processed_storage_key=proc_result["processed_storage_key"],
        processed_file_path=proc_result["processed_file_path"],
        blur_score=proc_result["blur_score"],
        brightness_score=proc_result["brightness_score"],
        contrast_score=proc_result["contrast_score"],
        quality_status=proc_result["quality_status"],
        quality_reasons=proc_result["quality_reasons"],
        original_dim=proc_result["original_dim"],
        processed_dim=proc_result["processed_dim"],
    )
    db.add(prep_log)
    await db.commit()
    await db.refresh(image_record)

    quality_schema = QualityAssessmentSchema(
        blur_score=proc_result["blur_score"],
        brightness_score=proc_result["brightness_score"],
        contrast_score=proc_result["contrast_score"],
        quality_status=proc_result["quality_status"],
        quality_reasons=proc_result["quality_reasons"],
        original_dim=proc_result["original_dim"],
        processed_dim=proc_result["processed_dim"],
    )

    img_schema = ImageRecordResponse(
        id=image_record.id,
        case_id=image_record.case_id,
        original_filename=image_record.original_filename,
        storage_key=image_record.storage_key,
        file_path=image_record.file_path,
        mime_type=image_record.mime_type,
        file_size_bytes=image_record.file_size_bytes,
        width=image_record.width,
        height=image_record.height,
        created_at=image_record.created_at,
        quality=quality_schema
    )

    return ImageUploadResponse(
        image=img_schema,
        preprocessing=quality_schema,
        message="Image successfully validated, quality-checked, and preprocessed."
    )


@router.get("/cases/{case_id}/images", response_model=List[ImageRecordResponse])
async def list_case_images(
    case: Case = Depends(get_case_for_user),
    db: AsyncSession = Depends(get_db)
):
    """List all image records associated with a case."""
    result = await db.execute(
        select(ImageRecord)
        .options(selectinload(ImageRecord.preprocessing_log))
        .where(ImageRecord.case_id == case.id)
        .order_by(ImageRecord.created_at.desc())
    )
    images = result.scalars().all()
    
    responses = []
    for img in images:
        q_schema = None
        if img.preprocessing_log:
            q_schema = QualityAssessmentSchema(
                blur_score=img.preprocessing_log.blur_score,
                brightness_score=img.preprocessing_log.brightness_score,
                contrast_score=img.preprocessing_log.contrast_score,
                quality_status=img.preprocessing_log.quality_status,
                quality_reasons=img.preprocessing_log.quality_reasons,
                original_dim=img.preprocessing_log.original_dim,
                processed_dim=img.preprocessing_log.processed_dim,
            )
        responses.append(ImageRecordResponse(
            id=img.id,
            case_id=img.case_id,
            original_filename=img.original_filename,
            storage_key=img.storage_key,
            file_path=img.file_path,
            mime_type=img.mime_type,
            file_size_bytes=img.file_size_bytes,
            width=img.width,
            height=img.height,
            created_at=img.created_at,
            quality=q_schema
        ))
    return responses


@router.get("/images/{image_id}/file")
async def get_original_image_file(image_id: str, db: AsyncSession = Depends(get_db)):
    """Serve the original raw uploaded image file."""
    result = await db.execute(select(ImageRecord).where(ImageRecord.id == image_id))
    img = result.scalar_one_or_none()
    if not img or not os.path.exists(img.file_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(img.file_path, media_type=img.mime_type)


@router.get("/images/{image_id}/preprocessed_file")
async def get_preprocessed_image_file(image_id: str, db: AsyncSession = Depends(get_db)):
    """Serve the standardized 224x224 preprocessed image file."""
    result = await db.execute(
        select(PreprocessingLog).where(PreprocessingLog.image_id == image_id)
    )
    prep = result.scalar_one_or_none()
    if not prep or not os.path.exists(prep.processed_file_path):
        raise HTTPException(status_code=404, detail="Preprocessed image file not found")
    return FileResponse(prep.processed_file_path, media_type="image/png")
