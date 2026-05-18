"""SQLAlchemy ORM models for the CXR Detection database schema.

Defines six tables:
- cxr_studies: Core study records tracking upload status and inference results.
- cxr_detections: Individual nodule detections with bounding box coordinates.
- cxr_annotated_images: Post-inference images with drawn bounding boxes.
- cxr_original_images: Raw uploaded images stored as binary blobs.
- cxr_validations: Radiologist validation of AI detection results.
- cxr_manual_annotations: Manual bounding box corrections by radiologists.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Enum, ForeignKey, LargeBinary, Index
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class CxrStudy(Base):
    """A chest X-ray study submitted for nodule detection.

    Tracks the full lifecycle from upload ("queued") through inference
    ("processing") to final result ("completed" or "error"). Cascades
    deletes to related detections and images.

    Attributes:
        study_uid: Unique external identifier (e.g. "CXR-A1B2C3D4E5F6").
        status: Current processing state (queued, processing, completed, error).
        num_detections: Count of nodules found by inference.
        inference_time_ms: Model inference duration in milliseconds.
    """

    __tablename__ = "cxr_studies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_uid = Column(String(100), unique=True, nullable=False, index=True)
    patient_id = Column(String(50), index=True)
    patient_name = Column(String(200))
    status = Column(Enum("queued", "processing", "completed", "error"), default="queued")
    image_format = Column(String(10), default="png")
    num_detections = Column(Integer, default=0)
    inference_time_ms = Column(Float)
    error_message = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime)

    detections = relationship("CxrDetection", back_populates="study", cascade="all, delete-orphan")
    annotated_image = relationship("CxrAnnotatedImage", back_populates="study", uselist=False, cascade="all, delete-orphan")
    original_image = relationship("CxrOriginalImage", back_populates="study", uselist=False, cascade="all, delete-orphan")
    validation = relationship("CxrValidation", back_populates="study", uselist=False, cascade="all, delete-orphan")


class CxrDetection(Base):
    """A single nodule detection within a CXR study.

    Stores bounding box coordinates (x1, y1, x2, y2) in pixel space,
    the confidence score, and which model produced the detection.

    Attributes:
        x1, y1: Top-left corner of the bounding box.
        x2, y2: Bottom-right corner of the bounding box.
        score: Model confidence score (0.0 to 1.0).
        label: Detection class label (default "nodule").
        model_source: Name of the model or ensemble that produced this detection.
    """

    __tablename__ = "cxr_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey("cxr_studies.id", ondelete="CASCADE"), nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    score = Column(Float, nullable=False)
    label = Column(String(50), default="nodule")
    model_source = Column(String(30))

    study = relationship("CxrStudy", back_populates="detections")


class CxrAnnotatedImage(Base):
    """Post-inference annotated image with bounding boxes drawn on the CXR.

    Stored as a binary PNG blob. One-to-one relationship with CxrStudy.
    """

    __tablename__ = "cxr_annotated_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey("cxr_studies.id", ondelete="CASCADE"), nullable=False, unique=True)
    image_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    study = relationship("CxrStudy", back_populates="annotated_image")


class CxrOriginalImage(Base):
    """Original uploaded CXR image stored as a binary blob.

    Preserves the raw file as uploaded by the user before any processing.
    One-to-one relationship with CxrStudy.
    """

    __tablename__ = "cxr_original_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey("cxr_studies.id", ondelete="CASCADE"), nullable=False, unique=True)
    image_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    study = relationship("CxrStudy", back_populates="original_image")


class CxrValidation(Base):
    """Radiologist validation of AI detection results.

    Records whether the radiologist agrees with the AI detections
    and any manual corrections made.

    Attributes:
        validated_by: Name or ID of the validating radiologist.
        validation_result: Assessment outcome (correct, incorrect, partial).
        notes: Free-text comments from the radiologist.
    """

    __tablename__ = "cxr_validations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(Integer, ForeignKey("cxr_studies.id", ondelete="CASCADE"), nullable=False, unique=True)
    validated_by = Column(String(100))
    validation_result = Column(Enum("correct", "incorrect", "partial"), nullable=False)
    notes = Column(Text)
    validated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    study = relationship("CxrStudy", back_populates="validation")
    manual_annotations = relationship("CxrManualAnnotation", back_populates="validation", cascade="all, delete-orphan")


class CxrManualAnnotation(Base):
    """Manual bounding box annotation by radiologist.

    Used when the AI missed a nodule or the radiologist wants to
    correct the detection. These annotations form the ground truth
    for prospective studies and model retraining.

    Attributes:
        x1, y1: Top-left corner of the bounding box.
        x2, y2: Bottom-right corner of the bounding box.
        label: Annotation class label (default "nodule").
        annotation_type: missed (AI missed it), false_positive (AI was wrong),
            or corrected (radiologist adjusted the bbox).
    """

    __tablename__ = "cxr_manual_annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    validation_id = Column(Integer, ForeignKey("cxr_validations.id", ondelete="CASCADE"), nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    label = Column(String(50), default="nodule")
    annotation_type = Column(Enum("missed", "false_positive", "corrected"), default="missed")
    notes = Column(Text)

    validation = relationship("CxrValidation", back_populates="manual_annotations")
