from typing import Any

from sqlalchemy.orm import Session

from app.models import TaskEvent


def record_event(
    db: Session,
    task_id: str,
    event_type: str,
    stage: str,
    payload: dict[str, Any] | None = None,
) -> TaskEvent:
    event = TaskEvent(
        task_id=task_id,
        event_type=event_type,
        stage=stage,
        payload=payload or {},
    )
    db.add(event)
    db.flush()
    return event
