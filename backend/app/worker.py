from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.engine.codelens import (
    IntegrationUnavailable,
    LocalizationRunError,
    run_localization,
)
from app.engine.repair import run_quick_repair
from app.engine.evistate import handoff
from app.engine.persist_evistate import persist_snapshot
from app.engine.path_resolution import resolve_candidate_paths
from app.events import record_event
from app.models import (
    CandidateHypothesis,
    Evidence,
    OutboxMessage,
    Task,
    TaskStage,
    ToolCall,
)

TERMINAL = {"blocked", "succeeded", "failed", "cancelled"}


def _event(
    db,
    task: Task,
    event_type: str,
    stage: str,
    payload: dict[str, Any] | None = None,
) -> None:
    record_event(db, task.id, event_type, stage, payload)


def _stage(db, task_id: str, name: str, status: str, detail=None) -> None:
    row = db.scalar(
        select(TaskStage).where(TaskStage.task_id == task_id, TaskStage.name == name)
    )
    now = datetime.now(timezone.utc)
    if row is None:
        row = TaskStage(task_id=task_id, name=name, status=status, detail=detail or {})
        db.add(row)
    else:
        row.status = status
        row.detail = detail or {}
    if status == "running" and row.started_at is None:
        row.started_at = now
    if status in {"blocked", "failed", "succeeded", "skipped"}:
        row.finished_at = now


def _block(db, task: Task, stage: str, message: str) -> None:
    task.status = "blocked"
    task.current_stage = stage
    task.error_message = message
    task.finished_at = datetime.now(timezone.utc)
    _stage(db, task.id, stage, "blocked", {"reason": message})
    _event(db, task, "stage_blocked", stage, {"reason": message})
    db.commit()


def _persist_candidates(db, task: Task, output) -> int:
    count = 0
    for rank, payload in enumerate(output.candidates, start=1):
        if not isinstance(payload, dict):
            continue
        candidate = CandidateHypothesis(
            task_id=task.id,
            rank=rank,
            target=str(payload.get("target") or payload.get("file_path") or ""),
            target_type=str(payload.get("target_type") or "file"),
            file_path=str(payload.get("file_path") or ""),
            function_name=str(payload.get("function_name") or ""),
            line_start=payload.get("line_start"),
            line_end=payload.get("line_end"),
            confidence=float(payload.get("confidence") or 0.0),
            hypothesis=str(payload.get("reason") or ""),
            source_payload=payload,
        )
        db.add(candidate)
        db.flush()
        reason = str(payload.get("reason") or "").strip()
        if reason:
            db.add(
                Evidence(
                    task_id=task.id,
                    candidate_id=candidate.id,
                    source="codelens",
                    kind="candidate_reason",
                    content=reason,
                    source_payload={"candidate_rank": rank},
                )
            )
        count += 1

    trajectory = output.policy_trajectory
    steps = (trajectory or {}).get("steps", []) if isinstance(trajectory, dict) else []
    for index, step in enumerate(steps, start=1):
        serialized = json.dumps(step, ensure_ascii=False, default=str)
        db.add(
            Evidence(
                task_id=task.id,
                source="codelens",
                kind="policy_transition",
                content=serialized,
                source_payload={"step": index},
            )
        )
    return count


@celery_app.task(name="app.worker.publish_outbox")
def publish_outbox() -> int:
    published = 0
    with SessionLocal() as db:
        pending = db.scalars(
            select(OutboxMessage)
            .where(OutboxMessage.published_at.is_(None))
            .order_by(OutboxMessage.created_at)
            .limit(50)
            .with_for_update(skip_locked=True)
        ).all()
        for message in pending:
            if message.topic == "run_localization":
                celery_app.send_task(
                    "app.worker.run_task",
                    args=[message.task_id],
                    task_id=message.id,
                )
            message.published_at = datetime.now(timezone.utc)
            published += 1
        db.commit()
    return published


@celery_app.task(
    bind=True,
    name="app.worker.run_task",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def run_task(self, task_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        task = db.scalar(select(Task).where(Task.id == task_id).with_for_update())
        if task is None or task.status in TERMINAL:
            return {"status": task.status if task else "missing"}
        if task.status == "running":
            return {"status": "already_running"}

        task.status = "running"
        task.current_stage = "localization"
        task.started_at = task.started_at or datetime.now(timezone.utc)
        task.error_message = None
        _stage(db, task.id, "localization", "running")
        _event(db, task, "stage_started", "localization", {"message": "Calling CodeLens localization pipeline."})
        tool_call = ToolCall(
            task_id=task.id,
            tool_name="CodeLens.LocalizationPipeline.run_sync",
            status="running",
            arguments={
                "repo_key": task.repo_key,
                "repository_path": task.repository_path,
                "repograph_path": task.repograph_path,
            },
        )
        db.add(tool_call)
        db.commit()

        try:
            output = run_localization(
                issue_text=task.issue_text,
                repo_key=task.repo_key,
                repository_path=task.repository_path,
                repograph_path=task.repograph_path,
                repository_root=settings.repository_root,
                repograph_root=settings.repograph_root,
                codelens_home=settings.codelens_home,
                vectorstore_root=settings.vectorstore_root,
            )
        except IntegrationUnavailable as exc:
            tool_call.status = "blocked"
            tool_call.error = str(exc)
            tool_call.finished_at = datetime.now(timezone.utc)
            db.commit()
            _block(db, task, "localization", str(exc))
            return {"status": "blocked", "reason": str(exc)}
        except LocalizationRunError as exc:
            tool_call.status = "retrying" if self.request.retries < self.max_retries else "failed"
            tool_call.error = str(exc)
            tool_call.finished_at = datetime.now(timezone.utc)
            if self.request.retries < self.max_retries:
                task.status = "queued"
                task.current_stage = "localization"
                _stage(db, task.id, "localization", "retrying", {"attempt": self.request.retries + 1})
                _event(
                    db,
                    task,
                    "retry_scheduled",
                    "localization",
                    {"attempt": self.request.retries + 1, "reason": str(exc)},
                )
                db.commit()
                raise self.retry(exc=exc)
            task.status = "failed"
            task.current_stage = "localization"
            task.error_message = str(exc)
            task.finished_at = datetime.now(timezone.utc)
            _stage(db, task.id, "localization", "failed", {"reason": str(exc)})
            _event(db, task, "task_failed", "localization", {"reason": str(exc)})
            db.commit()
            raise

        output.candidates, path_mappings = resolve_candidate_paths(
            output.candidates,
            Path(settings.repository_root) / task.repository_path,
        )
        if path_mappings:
            _event(db, task, "candidate_paths_resolved", "localization", {"mappings": path_mappings})
        tool_call.status = "succeeded"
        tool_call.result = {
            "candidate_count": len(output.candidates),
            "has_policy_trajectory": output.policy_trajectory is not None,
            "retrieval_count": len(output.retrieval_summary),
            "path_mappings": path_mappings,
        }
        tool_call.finished_at = datetime.now(timezone.utc)
        count = _persist_candidates(db, task, output)
        _stage(db, task.id, "localization", "succeeded", {"candidate_count": count})
        _event(
            db,
            task,
            "localization_completed",
            "localization",
            {"candidate_count": count, "raw_output": output.raw_output[:4000]},
        )

        task.current_stage = "evistate"
        _stage(db, task.id, "evistate", "running")
        _event(db, task, "stage_started", "evistate")
        db.commit()
        evi_call = ToolCall(task_id=task.id,tool_name="EviState.StateTracker+Barrier",status="running",arguments={"mapping":"repocoding-file-projection-v1"})
        db.add(evi_call)
        db.commit()
        try:
            evi = handoff(candidates=output.candidates,task_id=task.id,repo_key=task.repo_key,
                repository=Path(settings.repository_root)/task.repository_path,
                evistate_home=settings.evistate_home,artifact_root=settings.artifact_root)
            persist_snapshot(db,task,evi)
            evi_call.status = "succeeded"
            evi_call.result = evi
            evi_call.finished_at = datetime.now(timezone.utc)
            _stage(db,task.id,"evistate","succeeded",{"paths":evi["paths"],"ruleset":evi["ruleset"]})
            db.commit()
        except Exception as exc:
            db.rollback()
            evi_call = db.get(ToolCall,evi_call.id)
            evi_call.status = "blocked" if isinstance(exc,IntegrationUnavailable) else "failed"
            evi_call.error = str(exc)
            evi_call.finished_at = datetime.now(timezone.utc)
            _block(db,task,"evistate",str(exc))
            return {"status":"blocked","reason":str(exc)}
        repository = (Path(settings.repository_root) / task.repository_path).resolve()
        selected_path = next(
            (
                path for path in evi["paths"]
                if path.endswith(".java") and (repository / path).resolve().is_file()
                and repository in (repository / path).resolve().parents
            ),
            None,
        )
        if selected_path is None:
            reason = "CodeLens/EviState did not select an existing Java source file for repair."
            _block(db, task, "repair", reason)
            return {"status": "blocked", "reason": reason}

        task.current_stage = "repair"
        _stage(db, task.id, "repair", "running", {"selected_file": selected_path, "source": "codelens"})
        _event(db, task, "stage_started", "repair", {"selected_file": selected_path, "source": "codelens"})
        db.commit()
        try:
            patch = run_quick_repair(db, task, selected_path, settings, evi["code_version"])
        except IntegrationUnavailable as exc:
            _block(db, task, "repair", str(exc))
            return {"status": "blocked", "reason": str(exc)}
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            task.status = "failed"
            task.current_stage = "repair"
            task.error_message = reason
            task.finished_at = datetime.now(timezone.utc)
            _stage(db, task.id, "repair", "failed", {"reason": reason, "selected_file": selected_path})
            _stage(db, task.id, "tests", "skipped", {"reason": "Patch generation failed."})
            _event(db, task, "task_failed", "repair", {"reason": reason})
            db.commit()
            return {"status": "failed", "reason": reason}

        _stage(db, task.id, "repair", "succeeded", {**patch, "selected_file": selected_path})
        task.current_stage = "tests"
        _stage(db, task.id, "tests", "pending", {"reason": "Waiting for isolated Java validation."})
        _event(db, task, "tests_queued", "tests", {"selected_file": selected_path})
        db.commit()
        try:
            celery_app.send_task(
                "app.validation_worker.validate_patch",
                args=[task.id, selected_path],
                queue="validation",
            )
        except Exception as exc:
            reason = f"Could not queue Java validation: {type(exc).__name__}: {exc}"
            _block(db, task, "tests", reason)
            return {"status": "blocked", "reason": reason}
        return {"status": "running", "selected_file": selected_path, **patch}
