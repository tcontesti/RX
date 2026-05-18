"""Pydantic schemas for API request validation and response serialization.

These schemas define the shape of data exchanged between the CXR Detection
API and its clients. ORM models are converted to response schemas via
``from_attributes = True``.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Literal, Optional


class DetectionOut(BaseModel):
    """Response schema for a single nodule detection.

    Attributes:
        x1, y1: Top-left corner of bounding box (pixels).
        x2, y2: Bottom-right corner of bounding box (pixels).
        score: Confidence score from the detection model (0.0-1.0).
        label: Detection class (typically "nodule").
        model_source: Which model or ensemble produced this detection.
    """

    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    model_source: Optional[str] = None

    class Config:
        from_attributes = True


class StudyOut(BaseModel):
    """Full response schema for a CXR study, including nested detections.

    Attributes:
        study_uid: Unique external study identifier.
        status: Processing state (queued, processing, completed, error).
        num_detections: Total nodules detected.
        has_annotated_image: Whether an annotated image is available for download.
    """

    id: int
    study_uid: str
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    status: str
    num_detections: int
    inference_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    detections: list[DetectionOut] = []
    has_annotated_image: bool = False
    validation: Optional["ValidationOut"] = None

    class Config:
        from_attributes = True


class ManualAnnotationIn(BaseModel):
    """Input schema for a manual bounding box annotation."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "nodule"
    annotation_type: Literal["missed", "false_positive", "corrected"] = "missed"
    notes: Optional[str] = None


class ManualAnnotationOut(BaseModel):
    """Output schema for a manual annotation."""

    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    annotation_type: str
    notes: Optional[str] = None

    class Config:
        from_attributes = True


class ValidationIn(BaseModel):
    """Input schema for submitting a radiologist validation.

    Attributes:
        validation_result: Assessment (correct, incorrect, partial).
        validated_by: Name or ID of the radiologist.
        notes: Free-text comments.
        annotations: Manual bounding box corrections. May be empty
            when validation_result is "correct".
    """

    validation_result: Literal["correct", "incorrect", "partial"]
    validated_by: Optional[str] = None
    notes: Optional[str] = None
    annotations: list[ManualAnnotationIn] = []


class ValidationOut(BaseModel):
    """Output schema for a validation record with nested annotations."""

    id: int
    validation_result: str
    validated_by: Optional[str] = None
    notes: Optional[str] = None
    validated_at: datetime
    manual_annotations: list[ManualAnnotationOut] = []

    class Config:
        from_attributes = True


# Resolve forward reference to ValidationOut used in StudyOut
StudyOut.model_rebuild()


class UploadResponse(BaseModel):
    """Response returned after a successful CXR image upload.

    Attributes:
        study_uid: Assigned study identifier for tracking.
        status: Initial processing status (typically "processing").
        message: Human-readable confirmation message.
    """

    study_uid: str
    status: str
    message: str


class HistoryParams(BaseModel):
    """Query parameters for paginated study history listing.

    Attributes:
        page: Page number (1-indexed).
        per_page: Number of results per page.
        status: Optional filter by study status.
        patient_id: Optional filter by patient identifier.
    """

    page: int = 1
    per_page: int = 20
    status: Optional[str] = None
    patient_id: Optional[str] = None


class StatsOut(BaseModel):
    """Aggregate statistics across all CXR studies.

    Attributes:
        total_studies: Total number of studies in the database.
        completed: Number of studies with completed inference.
        with_nodules: Number of studies where at least one nodule was detected.
        avg_inference_ms: Average inference time across completed studies.
    """

    total_studies: int
    completed: int
    with_nodules: int
    avg_inference_ms: Optional[float] = None
