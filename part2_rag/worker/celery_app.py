import os
from celery import Celery

from ..logger import get_logger

logger = get_logger(__name__)

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

celery_app = Celery(
    "rag_worker",
    broker=broker_url,
    result_backend=result_backend,
    include=["part2_rag.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    task_soft_time_limit=300,
    worker_max_tasks_per_child=100,
    worker_prefetch_multiplier=1,
)

logger.info("Celery app configured: broker=%s backend=%s", broker_url, result_backend)
