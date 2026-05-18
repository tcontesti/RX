"""Health check router for monitoring API, database, and RabbitMQ status.

Returns an aggregate status of "ok" when all dependencies are reachable,
or "degraded" if any check fails.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import aio_pika
from app.config import settings
from app.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/api/health")
async def health(db: AsyncSession = Depends(get_db)):
    """Run liveness checks against all service dependencies.

    Checks MySQL connectivity via a simple SELECT and RabbitMQ via
    a short-timeout connection attempt.

    Args:
        db: Async database session.

    Returns:
        dict: Per-dependency status (ok/error) and overall status
            (ok or degraded).
    """
    checks = {"api": "ok", "database": "unknown", "rabbitmq": "unknown"}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"
    try:
        conn = await aio_pika.connect_robust(settings.RABBITMQ_URL, timeout=3)
        await conn.close()
        checks["rabbitmq"] = "ok"
    except Exception:
        checks["rabbitmq"] = "error"

    status = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {**checks, "status": status}
