from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app, get_stream_session_factory
from app.models import OutboxMessage, Task


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    test_sessions = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)

    def override_db():
        with test_sessions() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_stream_session_factory] = lambda: test_sessions
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    test_engine.dispose()


def create_payload() -> dict[str, str]:
    return {
        "issue_title": "Null value causes parsing error",
        "issue_text": "When the optional field is null, parsing raises an exception.",
        "repo_key": "sample__java",
        "repository_path": "sample__java",
        "repograph_path": "sample__java",
    }


def test_create_task_persists_task_event_and_outbox(client: TestClient) -> None:
    response = client.post("/api/tasks", json=create_payload())

    assert response.status_code == 202
    body = response.json()
    task_id = body["task"]["id"]
    assert body["task"]["status"] == "queued"
    assert body["event_id"] > 0

    events = client.get(f"/api/tasks/{task_id}/events").json()
    assert len(events) == 1
    assert events[0]["event_type"] == "task_created"
    assert events[0]["payload"]["message"]

    db_generator = app.dependency_overrides[get_db]()
    session = next(db_generator)
    try:
        outbox = session.scalar(select(OutboxMessage).where(OutboxMessage.task_id == task_id))
        assert outbox is not None
        assert outbox.topic == "run_localization"
    finally:
        db_generator.close()


def test_event_api_supports_cursor_replay(client: TestClient) -> None:
    created = client.post("/api/tasks", json=create_payload()).json()
    task_id = created["task"]["id"]

    first = client.get(f"/api/tasks/{task_id}/events?after=0").json()
    replay = client.get(f"/api/tasks/{task_id}/events?after={first[0]['id']}").json()

    assert len(first) == 1
    assert replay == []


def test_repository_paths_cannot_escape_mounted_roots(client: TestClient) -> None:
    payload = create_payload()
    payload["repository_path"] = "../outside"

    response = client.post("/api/tasks", json=payload)

    assert response.status_code == 422


def test_sse_replays_persisted_events_and_closes_terminal_stream(client: TestClient) -> None:
    created = client.post("/api/tasks", json=create_payload()).json()
    task_id = created["task"]["id"]
    db_generator = app.dependency_overrides[get_db]()
    session = next(db_generator)
    try:
        task = session.get(Task, task_id)
        assert task is not None
        task.status = "blocked"
        session.commit()
    finally:
        db_generator.close()

    response = client.get(f"/api/tasks/{task_id}/stream")

    assert response.status_code == 200
    assert "event: task_event" in response.text
    assert "task_created" in response.text
    assert "event: stream_end" in response.text
