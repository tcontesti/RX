"""Shared RabbitMQ helpers for the cxr-detection platform.

Provides connection management and message publishing utilities used by
both the API service and the inference pipeline. Queue names are defined
as module-level constants for consistency across producers and consumers.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any
import aio_pika

logger = logging.getLogger(__name__)

QUEUE_INFERENCE = "cxr.inference"
QUEUE_RESULTS = "cxr.results"


async def get_connection(rabbitmq_url: str) -> aio_pika.RobustConnection:
    """Create a robust (auto-reconnecting) RabbitMQ connection.

    Args:
        rabbitmq_url: AMQP connection URL (e.g. "amqp://user:pass@host:5672/").

    Returns:
        A RobustConnection that automatically reconnects on failure.
    """
    return await aio_pika.connect_robust(rabbitmq_url)


async def publish_inference_task(
    channel: aio_pika.Channel,
    study_uid: str,
    image_base64: str,
    image_format: str = "png",
) -> None:
    """Publish an inference task to the queue consumed by the Spark service.

    The message is persisted (delivery_mode=PERSISTENT) so it survives
    broker restarts.

    Args:
        channel: Open AMQP channel.
        study_uid: Unique study identifier to track the request.
        image_base64: Base64-encoded image data.
        image_format: Image file format without dot (e.g. "png", "dcm").
    """
    queue = await channel.declare_queue(QUEUE_INFERENCE, durable=True)
    body = {
        "study_id": study_uid,
        "image_data": image_base64,
        "format": image_format,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    message = aio_pika.Message(
        body=json.dumps(body).encode(),
        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        content_type="application/json",
    )
    await channel.default_exchange.publish(message, routing_key=QUEUE_INFERENCE)
    logger.info(f"Task published: study_uid={study_uid}")
