"""Router for retrieving, downloading, and deleting CXR study results.

Provides endpoints to fetch study metadata with detections, download
annotated/original images, and delete individual or all studies.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.models.models import CxrStudy, CxrAnnotatedImage, CxrOriginalImage, CxrValidation
from app.schemas.schemas import StudyOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cxr", tags=["CXR"])


@router.get("/results/{study_uid}", response_model=StudyOut)
async def get_results(study_uid: str, db: AsyncSession = Depends(get_db)):
    """Retrieve study results including all detections.

    Args:
        study_uid: Unique study identifier.
        db: Async database session.

    Returns:
        StudyOut with detections and annotated image availability flag.

    Raises:
        HTTPException 404: Study not found.
    """
    result = await db.execute(
        select(CxrStudy)
        .options(selectinload(CxrStudy.detections))
        .options(selectinload(CxrStudy.annotated_image))
        .options(selectinload(CxrStudy.validation).selectinload(CxrValidation.manual_annotations))
        .where(CxrStudy.study_uid == study_uid)
    )
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(404, "Estudio no encontrado")

    out = StudyOut.model_validate(study)
    out.has_annotated_image = study.annotated_image is not None
    return out


@router.get("/results/{study_uid}/image")
async def get_annotated_image(study_uid: str, db: AsyncSession = Depends(get_db)):
    """Download the annotated image (with bounding boxes) as PNG.

    Args:
        study_uid: Unique study identifier.
        db: Async database session.

    Returns:
        Raw PNG image response.

    Raises:
        HTTPException 404: Annotated image not found.
    """
    result = await db.execute(
        select(CxrAnnotatedImage).join(CxrStudy).where(CxrStudy.study_uid == study_uid)
    )
    img = result.scalar_one_or_none()
    if not img:
        raise HTTPException(404, "Imagen anotada no encontrada")
    return Response(content=img.image_data, media_type="image/png")


@router.get("/results/{study_uid}/original")
async def get_original_image(study_uid: str, db: AsyncSession = Depends(get_db)):
    """Download the original uploaded CXR image in its native format.

    Args:
        study_uid: Unique study identifier.
        db: Async database session.

    Returns:
        Raw image response with correct Content-Type.

    Raises:
        HTTPException 404: Original image not found.
    """
    result = await db.execute(
        select(CxrOriginalImage, CxrStudy.image_format)
        .join(CxrStudy)
        .where(CxrStudy.study_uid == study_uid)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(404, "Imagen original no encontrada")
    img, fmt = row
    media_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                   "dcm": "application/dicom", "mha": "application/octet-stream"}
    return Response(content=img.image_data, media_type=media_types.get(fmt, "application/octet-stream"))


@router.delete("/results/{study_uid}")
async def delete_study(study_uid: str, db: AsyncSession = Depends(get_db)):
    """Delete a single study and all related detections and images.

    Cascade delete removes associated CxrDetection, CxrAnnotatedImage,
    and CxrOriginalImage records.

    Args:
        study_uid: Unique study identifier.
        db: Async database session.

    Returns:
        dict: Confirmation with the deleted study_uid.

    Raises:
        HTTPException 404: Study not found.
    """
    result = await db.execute(select(CxrStudy).where(CxrStudy.study_uid == study_uid))
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(404, "Estudio no encontrado")
    await db.delete(study)
    await db.commit()
    return {"deleted": study_uid}


@router.delete("/all")
async def delete_all_studies(db: AsyncSession = Depends(get_db)):
    """Delete all studies and their associated data.

    Uses bulk SQL DELETE. Child tables cascade via ON DELETE CASCADE in the schema.

    Args:
        db: Async database session.

    Returns:
        dict: Count of deleted studies.
    """
    from app.models.models import CxrDetection, CxrAnnotatedImage, CxrOriginalImage, CxrValidation, CxrManualAnnotation
    count = await db.scalar(select(func.count(CxrStudy.id)))
    # Delete children first (bulk), then studies — avoids loading all into memory
    await db.execute(delete(CxrManualAnnotation))
    await db.execute(delete(CxrValidation))
    await db.execute(delete(CxrDetection))
    await db.execute(delete(CxrAnnotatedImage))
    await db.execute(delete(CxrOriginalImage))
    await db.execute(delete(CxrStudy))
    await db.commit()
    return {"deleted": count or 0}
