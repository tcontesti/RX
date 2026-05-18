"""Router for radiologist validation and manual annotation of AI results.

Provides endpoints to submit validations, retrieve validation status,
and export validated datasets for model retraining.
"""

import csv
import io
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.models.models import CxrStudy, CxrValidation, CxrManualAnnotation
from app.schemas.schemas import ValidationIn, ValidationOut

router = APIRouter(prefix="/api/cxr", tags=["Validation"])


@router.post("/results/{study_uid}/validate", response_model=ValidationOut)
async def validate_study(study_uid: str, body: ValidationIn, db: AsyncSession = Depends(get_db)):
    """Submit radiologist validation for a completed study.

    Creates a validation record and optional manual annotations.
    If the study was already validated, overwrites the previous validation.

    Args:
        study_uid: Unique study identifier.
        body: Validation data including result, optional annotations.
        db: Async database session.

    Returns:
        ValidationOut with the created validation and annotations.

    Raises:
        HTTPException 404: Study not found.
        HTTPException 400: Study not in completed status.
    """
    result = await db.execute(
        select(CxrStudy)
        .options(selectinload(CxrStudy.validation))
        .where(CxrStudy.study_uid == study_uid)
    )
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(404, "Estudio no encontrado")
    if study.status != "completed":
        raise HTTPException(400, "Solo se pueden validar estudios completados")

    # Overwrite existing validation if present
    if study.validation:
        await db.delete(study.validation)
        await db.flush()

    validation = CxrValidation(
        study_id=study.id,
        validated_by=body.validated_by,
        validation_result=body.validation_result,
        notes=body.notes,
    )
    db.add(validation)
    await db.flush()

    for ann in body.annotations:
        manual = CxrManualAnnotation(
            validation_id=validation.id,
            x1=ann.x1, y1=ann.y1, x2=ann.x2, y2=ann.y2,
            label=ann.label,
            annotation_type=ann.annotation_type,
            notes=ann.notes,
        )
        db.add(manual)

    await db.commit()
    await db.refresh(validation)

    # Reload with annotations
    result2 = await db.execute(
        select(CxrValidation)
        .options(selectinload(CxrValidation.manual_annotations))
        .where(CxrValidation.id == validation.id)
    )
    validation = result2.scalar_one()
    return ValidationOut.model_validate(validation)


@router.get("/results/{study_uid}/validation", response_model=ValidationOut)
async def get_validation(study_uid: str, db: AsyncSession = Depends(get_db)):
    """Get validation status for a study.

    Args:
        study_uid: Unique study identifier.
        db: Async database session.

    Returns:
        ValidationOut with nested manual annotations.

    Raises:
        HTTPException 404: Validation not found.
    """
    result = await db.execute(
        select(CxrValidation)
        .join(CxrStudy)
        .options(selectinload(CxrValidation.manual_annotations))
        .where(CxrStudy.study_uid == study_uid)
    )
    validation = result.scalar_one_or_none()
    if not validation:
        raise HTTPException(404, "Validacion no encontrada")
    return ValidationOut.model_validate(validation)


@router.get("/validation/stats")
async def validation_stats(db: AsyncSession = Depends(get_db)):
    """Get validation statistics for prospective study reporting.

    Args:
        db: Async database session.

    Returns:
        dict: Counts of validated studies by result type, AI accuracy,
            and manual annotation breakdown.
    """
    total_completed = await db.scalar(
        select(func.count(CxrStudy.id)).where(CxrStudy.status == "completed")
    )
    total_validated = await db.scalar(select(func.count(CxrValidation.id)))
    correct = await db.scalar(
        select(func.count(CxrValidation.id)).where(CxrValidation.validation_result == "correct")
    )
    incorrect = await db.scalar(
        select(func.count(CxrValidation.id)).where(CxrValidation.validation_result == "incorrect")
    )
    partial = await db.scalar(
        select(func.count(CxrValidation.id)).where(CxrValidation.validation_result == "partial")
    )
    total_manual = await db.scalar(select(func.count(CxrManualAnnotation.id)))
    missed = await db.scalar(
        select(func.count(CxrManualAnnotation.id)).where(CxrManualAnnotation.annotation_type == "missed")
    )
    false_pos = await db.scalar(
        select(func.count(CxrManualAnnotation.id)).where(CxrManualAnnotation.annotation_type == "false_positive")
    )

    return {
        "total_completed": total_completed or 0,
        "total_validated": total_validated or 0,
        "pending_validation": (total_completed or 0) - (total_validated or 0),
        "correct": correct or 0,
        "incorrect": incorrect or 0,
        "partial": partial or 0,
        "accuracy": round(correct / total_validated, 4) if total_validated else None,
        "total_manual_annotations": total_manual or 0,
        "missed_nodules": missed or 0,
        "false_positives": false_pos or 0,
    }


@router.get("/validation/export")
async def export_dataset(
    format: str = Query("csv", enum=["csv", "json"]),
    include_ai: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """Export validated dataset for model retraining.

    Generates a CSV or JSON with all completed studies, including
    AI detections and manual corrections. Useful for creating
    ground truth datasets.

    Args:
        format: Output format ("csv" or "json").
        include_ai: Whether to include AI detections alongside manual annotations.
        db: Async database session.

    Returns:
        StreamingResponse (CSV) or list[dict] (JSON).
    """
    result = await db.execute(
        select(CxrStudy)
        .options(selectinload(CxrStudy.detections))
        .options(selectinload(CxrStudy.validation).selectinload(CxrValidation.manual_annotations))
        .where(CxrStudy.status == "completed")
        .order_by(CxrStudy.created_at)
    )
    studies = result.scalars().all()

    rows = []
    for s in studies:
        base = {
            "study_uid": s.study_uid,
            "patient_id": s.patient_id or "",
            "validation_result": s.validation.validation_result if s.validation else "pending",
            "validated_by": s.validation.validated_by if s.validation else "",
        }

        if include_ai:
            for d in s.detections:
                rows.append({**base, "source": "ai", "x": d.x1, "y": d.y1,
                             "width": d.x2 - d.x1, "height": d.y2 - d.y1,
                             "score": d.score, "label": d.label, "type": "detection"})

        if s.validation:
            for a in s.validation.manual_annotations:
                rows.append({**base, "source": "radiologist", "x": a.x1, "y": a.y1,
                             "width": a.x2 - a.x1, "height": a.y2 - a.y1,
                             "score": 1.0, "label": a.label, "type": a.annotation_type})

        # Include a row for studies with no detections and no annotations
        if not s.detections and (not s.validation or not s.validation.manual_annotations):
            rows.append({**base, "source": "none", "x": 0, "y": 0,
                         "width": 0, "height": 0, "score": 0, "label": "none", "type": "negative"})

    if format == "json":
        return rows

    fieldnames = ["study_uid", "patient_id", "validation_result", "validated_by",
                   "source", "x", "y", "width", "height", "score", "label", "type"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    if rows:
        writer.writerows(rows)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cxr_validated_dataset.csv"}
    )
