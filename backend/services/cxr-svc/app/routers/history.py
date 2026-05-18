"""Router for paginated study history and aggregate statistics.

Provides a filterable, searchable listing of past studies and a summary
statistics endpoint for dashboard use.
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload
from app.database import get_db
from app.models.models import CxrStudy, CxrValidation
from app.schemas.schemas import StudyOut, StatsOut

router = APIRouter(prefix="/api/cxr", tags=["CXR"])


@router.get("/history", response_model=list[StudyOut])
async def get_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: str = Query(None),
    patient_id: str = Query(None),
    search: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List studies with pagination, filtering, and free-text search.

    Results are ordered by creation date (newest first). The search
    parameter matches against study_uid, patient_id, and patient_name.

    Args:
        page: Page number (1-indexed).
        per_page: Results per page (1-100).
        status: Optional filter by study status.
        patient_id: Optional filter by patient identifier.
        search: Optional free-text search across uid, patient_id, patient_name.
        db: Async database session.

    Returns:
        list[StudyOut]: Paginated list of studies with detections.
    """
    query = (
        select(CxrStudy)
        .options(selectinload(CxrStudy.detections))
        .options(selectinload(CxrStudy.annotated_image))
        .options(selectinload(CxrStudy.validation).selectinload(CxrValidation.manual_annotations))
        .order_by(CxrStudy.created_at.desc())
    )
    if status:
        query = query.where(CxrStudy.status == status)
    if patient_id:
        query = query.where(CxrStudy.patient_id == patient_id)
    if search:
        query = query.where(
            or_(
                CxrStudy.study_uid.ilike(f"%{search}%"),
                CxrStudy.patient_id.ilike(f"%{search}%"),
                CxrStudy.patient_name.ilike(f"%{search}%"),
            )
        )

    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    studies = result.scalars().all()
    out = []
    for s in studies:
        item = StudyOut.model_validate(s)
        item.has_annotated_image = s.annotated_image is not None
        out.append(item)
    return out


@router.get("/stats", response_model=StatsOut)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """Return aggregate statistics across all studies.

    Args:
        db: Async database session.

    Returns:
        StatsOut: Total studies, completed count, studies with nodules,
            and average inference time.
    """
    total = await db.scalar(select(func.count(CxrStudy.id)))
    completed = await db.scalar(select(func.count(CxrStudy.id)).where(CxrStudy.status == "completed"))
    with_nodules = await db.scalar(select(func.count(CxrStudy.id)).where(CxrStudy.num_detections > 0))
    avg_time = await db.scalar(select(func.avg(CxrStudy.inference_time_ms)).where(CxrStudy.status == "completed"))
    return StatsOut(total_studies=total or 0, completed=completed or 0, with_nodules=with_nodules or 0, avg_inference_ms=avg_time)
