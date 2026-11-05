from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

JsonPayload = JSON().with_variant(JSONB, "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    issue_title: Mapped[str] = mapped_column(String(240), nullable=False)
    issue_text: Mapped[str] = mapped_column(Text, nullable=False)
    repo_key: Mapped[str] = mapped_column(String(160), nullable=False)
    repository_path: Mapped[str] = mapped_column(String(500), nullable=False)
    repograph_path: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    current_stage: Mapped[str] = mapped_column(String(40), nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text)
    # Q (workflow) and Z (cognitive) are separate EviState-owned snapshots.
    # They remain NULL until returned by an audited EviState adapter.
    workflow_state_q: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload)
    cognitive_state_z: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    stages: Mapped[list["TaskStage"]] = relationship(cascade="all, delete-orphan")
    candidates: Mapped[list["CandidateHypothesis"]] = relationship(cascade="all, delete-orphan")
    evidence: Mapped[list["Evidence"]] = relationship(cascade="all, delete-orphan")
    events: Mapped[list["TaskEvent"]] = relationship(cascade="all, delete-orphan")
    patches: Mapped[list["PatchArtifact"]] = relationship(cascade="all, delete-orphan")
    test_results: Mapped[list["TestResult"]] = relationship(cascade="all, delete-orphan")
    tool_calls: Mapped[list["ToolCall"]] = relationship(cascade="all, delete-orphan")


class TaskStage(Base):
    __tablename__ = "task_stages"
    __table_args__ = (UniqueConstraint("task_id", "name", name="uq_task_stage_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    detail: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CandidateHypothesis(Base):
    __tablename__ = "candidate_hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    target: Mapped[str] = mapped_column(Text, nullable=False)
    target_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, default="")
    function_name: Mapped[str] = mapped_column(Text, default="")
    line_start: Mapped[int | None] = mapped_column(Integer)
    line_end: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    source_payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    bindings: Mapped[list["SupportAttackBinding"]] = relationship(
        cascade="all, delete-orphan"
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("candidate_hypotheses.id", ondelete="CASCADE"), index=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SupportAttackBinding(Base):
    __tablename__ = "support_attack_bindings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("candidate_hypotheses.id", ondelete="CASCADE"), index=True
    )
    evidence_id: Mapped[str] = mapped_column(
        ForeignKey("evidence.id", ondelete="CASCADE"), index=True
    )
    relation: Mapped[str] = mapped_column(String(16), nullable=False)
    source_payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    __table_args__ = (
        Index("ix_binding_candidate_relation", "candidate_id", "relation"),
    )


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    tool_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JsonPayload)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PatchArtifact(Base):
    __tablename__ = "patch_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    diff_text: Mapped[str] = mapped_column(Text, nullable=False)
    worktree_path: Mapped[str] = mapped_column(String(500), nullable=False)
    source_payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TestResult(Base):
    __tablename__ = "test_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    build_tool: Mapped[str] = mapped_column(String(24), nullable=False)
    command: Mapped[list[str]] = mapped_column(JsonPayload, nullable=False)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    passed: Mapped[bool | None] = mapped_column()
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TaskEvent(Base):
    __tablename__ = "task_events"
    __table_args__ = (Index("ix_event_task_id_id", "task_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(40), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class OutboxMessage(Base):
    __tablename__ = "outbox_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), index=True)
    topic: Mapped[str] = mapped_column(String(120), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonPayload, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
