from celery import Celery
from celery.schedules import schedule

from app.config import settings

celery_app = Celery(
    "repocoding",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker", "app.validation_worker"],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "publish-task-outbox": {
            "task": "app.worker.publish_outbox",
            "schedule": schedule(run_every=1.0),
        }
    },
    timezone="UTC",
)
