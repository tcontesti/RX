"""CXR Detection API — FastAPI application entry point.

Initializes the FastAPI app with CORS middleware, registers all routers
(upload, results, history, health), and manages the application lifespan
including database engine disposal on shutdown.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.database import engine
from app.routers import upload, results, history, health, validation

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle.

    On startup, logs an informational message. On shutdown, disposes the
    SQLAlchemy async engine to close all pooled database connections.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application for the duration of its lifetime.
    """
    logging.getLogger("cxr").info("CXR Detection API starting...")
    yield
    await engine.dispose()
    logging.getLogger("cxr").info("CXR Detection API shutdown complete.")


app = FastAPI(
    title="CXR Detection API",
    description="API para deteccion de nodulos pulmonares en radiografias de torax",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a safe JSON error in production."""
    logger = logging.getLogger("cxr")
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    detail = str(exc) if settings.ENVIRONMENT == "development" else "Error interno del servidor"
    return JSONResponse(status_code=500, content={"detail": detail})


app.include_router(upload.router)
app.include_router(results.router)
app.include_router(history.router)
app.include_router(health.router)
app.include_router(validation.router)


@app.get("/")
async def root():
    """Return basic service identification and version.

    Returns:
        dict: Service name and version string.
    """
    return {"service": "CXR Detection API", "version": "1.0.0"}
