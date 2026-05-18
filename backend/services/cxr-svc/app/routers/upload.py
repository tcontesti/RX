"""Router for CXR image upload and inference task submission.

Handles file validation, database persistence of the original image,
and publishing an inference task to RabbitMQ for async processing.
"""

import base64
import json
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.models.models import CxrStudy, CxrOriginalImage
from app.schemas.schemas import UploadResponse
from app.config import settings
import aio_pika

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cxr", tags=["CXR"])


@router.post("/upload", response_model=UploadResponse)
async def upload_cxr(
    file: UploadFile = File(...),
    patient_id: str = Form(default=None),
    patient_name: str = Form(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a CXR image for nodule detection analysis.

    Accepts PNG, JPG, DICOM (.dcm), and MHA formats. Max 50MB.
    Stores original image in MySQL and publishes inference task to RabbitMQ.

    Args:
        file: Uploaded image file (multipart form).
        patient_id: Optional patient identifier.
        patient_name: Optional patient display name.
        db: Async database session.

    Returns:
        UploadResponse with study_uid and status.

    Raises:
        HTTPException 400: Unsupported file format.
        HTTPException 413: File exceeds size limit.
        HTTPException 503: RabbitMQ unavailable.
    """
    # Validate file format
    allowed = {".png", ".jpg", ".jpeg", ".dcm", ".mha"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(400, f"Formato no soportado: {ext}. Usar: {allowed}")

    # Validate file size
    contents = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(413, f"Archivo demasiado grande (max {settings.MAX_UPLOAD_SIZE_MB}MB)")

    study_uid = f"CXR-{uuid.uuid4().hex[:12].upper()}"
    fmt = ext.replace(".", "")

    # Step 1: Persist study + original image as "queued" (safe commit)
    study = CxrStudy(
        study_uid=study_uid,
        patient_id=patient_id,
        patient_name=patient_name,
        status="queued",
        image_format=fmt,
    )
    db.add(study)
    await db.flush()  # get study.id without committing

    original = CxrOriginalImage(study_id=study.id, image_data=contents)
    db.add(original)
    await db.commit()  # study + image persisted — safe to publish now

    # Step 2: Publish inference task to RabbitMQ
    try:
        connection = await aio_pika.connect_robust(settings.RABBITMQ_URL)
        async with connection:
            channel = await connection.channel()
            await channel.declare_queue("cxr.inference", durable=True)

            image_b64 = base64.b64encode(contents).decode("utf-8")
            body = {
                "study_id": study_uid,
                "image_data": image_b64,
                "format": fmt,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            message = aio_pika.Message(
                body=json.dumps(body).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )
            await channel.default_exchange.publish(message, routing_key="cxr.inference")

        # Step 3: Update status to "processing" after successful publish
        study.status = "processing"
        await db.commit()

        logger.info(f"Study {study_uid} queued for inference")

    except Exception as e:
        # Study stays as "queued" in DB — can be retried or cleaned up
        logger.error(f"Failed to queue study {study_uid}: {e}")
        raise HTTPException(503, "Servicio de inferencia no disponible")

    return UploadResponse(study_uid=study_uid, status="processing", message="Imagen enviada para análisis")
