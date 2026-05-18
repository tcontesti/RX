"""Result Consumer -- reads inference results from RabbitMQ and updates MySQL.

Runs as a standalone process that listens on the ``cxr.results`` queue.
For each message it updates the corresponding CxrStudy record with
detection results, annotated images, or error information.

Usage::

    python -m app.consumers.result_consumer
"""

import asyncio
import base64
import json
import logging
import os
import sys
from datetime import datetime, timezone

import aio_pika
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("result_consumer")

DB_URL = os.getenv("DB_URL", "").replace("pymysql", "aiomysql")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")

# Adjust sys.path so models can be imported when running as __main__
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.models import Base, CxrStudy, CxrDetection, CxrAnnotatedImage


async def process_result(message: aio_pika.IncomingMessage, session_maker):
    """Process a single inference result message from RabbitMQ.

    Parses the JSON payload, looks up the study in the database, and
    either saves detection results (on success) or records the error
    message (on failure). On DB errors the message is nacked and
    requeued so it is not lost.

    Args:
        message: Incoming AMQP message with JSON body containing
            study_id, status, detections, and optional annotated_image_base64.
        session_maker: SQLAlchemy async session factory.
    """
    try:
        result = json.loads(message.body)
        study_uid = result.get("study_id")
        status = result.get("status", "error")

        logger.info(f"Result received: study_uid={study_uid}, status={status}")

        async with session_maker() as db:
            stmt = select(CxrStudy).where(CxrStudy.study_uid == study_uid)
            row = await db.execute(stmt)
            study = row.scalar_one_or_none()

            if not study:
                logger.warning(f"Study {study_uid} not found in DB — acking to discard")
                await message.ack()
                return

            # Idempotency: skip if already in a terminal state
            if study.status in ("completed", "error"):
                logger.info(f"Study {study_uid} already {study.status} — acking duplicate")
                await message.ack()
                return

            if status == "completed":
                study.status = "completed"
                study.num_detections = result.get("num_detections", 0)
                study.inference_time_ms = result.get("inference_time_ms")
                study.completed_at = datetime.now(timezone.utc)

                for det in result.get("detections", []):
                    detection = CxrDetection(
                        study_id=study.id,
                        x1=det["x1"], y1=det["y1"],
                        x2=det["x2"], y2=det["y2"],
                        score=det["score"],
                        label=det.get("label", "nodule"),
                        model_source=det.get("model_source", "ensemble"),
                    )
                    db.add(detection)

                if result.get("annotated_image_base64"):
                    img_bytes = base64.b64decode(result["annotated_image_base64"])
                    annotated = CxrAnnotatedImage(study_id=study.id, image_data=img_bytes)
                    db.add(annotated)

                logger.info(f"Study {study_uid}: {study.num_detections} detections, {study.inference_time_ms}ms")

            elif status == "error":
                study.status = "error"
                study.error_message = result.get("error", "Unknown error")
                study.completed_at = datetime.now(timezone.utc)
                logger.error(f"Study {study_uid} error: {study.error_message}")

            await db.commit()

        await message.ack()

    except json.JSONDecodeError as e:
        logger.error(f"Malformed message body, discarding: {e}")
        await message.reject(requeue=False)

    except Exception as e:
        logger.error(f"Error processing result (will requeue): {e}", exc_info=True)
        await message.nack(requeue=True)
        await asyncio.sleep(5)  # back-off before next retry


async def main():
    """Entry point: connect to MySQL and RabbitMQ, then consume results indefinitely."""
    logger.info("Result Consumer starting...")

    engine = create_async_engine(DB_URL, echo=False, pool_recycle=300, pool_pre_ping=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    connection = await aio_pika.connect_robust(RABBITMQ_URL)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=10)

    queue = await channel.declare_queue("cxr.results", durable=True)

    logger.info("Consuming from cxr.results...")

    async with queue.iterator(no_ack=False) as queue_iter:
        async for message in queue_iter:
            await process_result(message, session_maker)


if __name__ == "__main__":
    asyncio.run(main())
