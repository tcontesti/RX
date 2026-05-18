"""Async database engine and session factory for MySQL via aiomysql.

Converts the pymysql connection URL from settings to aiomysql for async
support. Provides a FastAPI dependency (get_db) that yields scoped sessions.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import settings

# Replace sync driver with async driver for SQLAlchemy async engine
db_url = settings.DB_URL.replace("pymysql", "aiomysql")
engine = create_async_engine(
    db_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
    pool_pre_ping=True,
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    """FastAPI dependency that provides an async database session.

    Yields:
        AsyncSession: A scoped SQLAlchemy async session. The session is
            automatically closed when the request completes.
    """
    async with async_session() as session:
        yield session
