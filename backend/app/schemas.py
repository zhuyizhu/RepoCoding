from datetime import datetime
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _validate_relative_path(value: str) -> str:
    cleaned = value.replace("\\", "/").strip()
    parts = cleaned.split("/")
    if (
        not cleaned
        or cleaned.startswith("/")
        or re.match(r"^[A-Za-z]:", cleaned)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("Path must be a non-empty path relative to the configured volume.")
    return cleaned


class TaskCreate(BaseModel):
    issue_title: str = Field(min_length=3, max_length=240)
    issue_text: str = Field(min_length=10, max_length=30000)
    repo_key: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$",
    )
    repository_path: str = Field(description="Path under REPOSITORY_ROOT.")
    repograph_path: str = Field(description="Path under REPOGRAPH_ROOT.")

    @field_validator("repository_path", "repograph_path")
    @classmethod
    def relative_to_mounted_root(cls, value: str) -> str:
        return _validate_relative_path(value)


class LocalDemoCreate(TaskCreate):
    file_path: str = Field(description="Repository-relative Java source file to inspect without a model.")
    repair_with_model: bool = False

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        path = _validate_relative_path(value)
        if not path.endswith(".java"):
            raise ValueError("Local demo file must be a Java source path.")
        return path


class TaskSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    issue_title: str
    repo_key: str
    status: str
    current_stage: str
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class BindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    evidence_id: str
    relation: str
    source_payload: dict[str, Any]


class CandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    rank: int
    target: str
    target_type: str
    file_path: str
    function_name: str
    line_start: int | None
    line_end: int | None
    confidence: float
    hypothesis: str
    source_payload: dict[str, Any]
    bindings: list[BindingOut] = Field(default_factory=list)


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    candidate_id: str | None
    source: str
    kind: str
    content: str
    source_payload: dict[str, Any]


class StageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    status: str
    detail: dict[str, Any]
    started_at: datetime | None
    finished_at: datetime | None


class PatchOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    diff_text: str
    worktree_path: str
    source_payload: dict[str, Any]
    created_at: datetime


class TestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    build_tool: str
    command: list[str]
    exit_code: int | None
    passed: bool | None
    stdout: str
    stderr: str
    started_at: datetime | None
    finished_at: datetime | None


class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    stage: str
    payload: dict[str, Any]
    created_at: datetime


class ToolCallOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    tool_name: str
    status: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class TaskDetail(TaskSummary):
    issue_text: str
    repository_path: str
    repograph_path: str
    workflow_state_q: dict[str, Any] | None
    cognitive_state_z: dict[str, Any] | None
    stages: list[StageOut] = Field(default_factory=list)
    candidates: list[CandidateOut] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    patches: list[PatchOut] = Field(default_factory=list)
    test_results: list[TestResultOut] = Field(default_factory=list)
    tool_calls: list[ToolCallOut] = Field(default_factory=list)


class TaskCreated(BaseModel):
    task: TaskSummary
    event_id: int


class HealthOut(BaseModel):
    status: Literal["ok"]
    integrations: dict[str, str]
