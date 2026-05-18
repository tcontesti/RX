"""Application configuration loaded from environment variables.

All settings fall back to sensible development defaults when the
corresponding environment variable is not set.
"""

import os


class Settings:
    """Central configuration for the CXR Detection API.

    Attributes:
        DB_URL: SQLAlchemy connection string for MySQL (pymysql driver).
        RABBITMQ_URL: AMQP connection URL for RabbitMQ.
        ENVIRONMENT: Runtime environment name (development, staging, production).
        LOG_LEVEL: Python logging level name (DEBUG, INFO, WARNING, ERROR).
        CORS_ORIGINS: Comma-separated list of allowed CORS origins.
        MAX_UPLOAD_SIZE_MB: Maximum allowed upload file size in megabytes.
    """

    DB_URL: str = os.getenv("DB_URL", "mysql+pymysql://cxr_app:CHANGE_ME@localhost:3306/cxr_detection")
    RABBITMQ_URL: str = os.getenv("RABBITMQ_URL", "amqp://cxr_worker:CHANGE_ME@localhost:5672/")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "http://localhost:5175").split(",")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))


settings = Settings()
