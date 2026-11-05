"""Persist the outcome of an isolated Java test run for a generated patch."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.celery_app import celery_app
from app.config import settings
from app.database import SessionLocal
from app.events import record_event
from app.models import PatchArtifact, Task, TaskStage, TestResult, ToolCall
from app.sandbox_client import run_java_tests


def now():
    return datetime.now(timezone.utc)


@celery_app.task(name="app.validation_worker.validate_patch")
def validate_patch(task_id: str, source_file: str) -> dict:
    with SessionLocal() as db:
        task = db.get(Task, task_id)
        if task is None:
            return {"status": "missing"}
        stage = db.query(TaskStage).filter_by(task_id=task_id, name="tests").one_or_none()
        if stage is None or stage.status in {"succeeded", "failed", "blocked"}:
            return {"status": "already_finished"}
        if db.query(TestResult).filter_by(task_id=task_id).first() is not None:
            return {"status": "already_recorded"}
        worktree = Path(settings.worktree_root).resolve() / task_id
        patch = db.query(PatchArtifact).filter_by(task_id=task_id).order_by(PatchArtifact.created_at.desc()).first()
        if patch is None or not worktree.is_dir():
            reason = "Patch worktree is unavailable for Java validation."
            stage.status = "blocked"; stage.detail = {"reason": reason}; stage.finished_at = now()
            task.status = "blocked"; task.current_stage = "tests"; task.error_message = reason; task.finished_at = now()
            record_event(db, task_id, "tests_blocked", "tests", {"reason": reason})
            db.commit()
            return {"status": "blocked", "reason": reason}
        tool = "maven" if (worktree / "pom.xml").is_file() else "gradle" if any((worktree / name).is_file() for name in ("build.gradle", "build.gradle.kts")) else None
        if tool is None:
            reason = "No Maven or Gradle build file in the patch worktree."
            stage.status = "blocked"; stage.detail = {"reason": reason}; stage.finished_at = now()
            task.status = "blocked"; task.current_stage = "tests"; task.error_message = reason; task.finished_at = now()
            record_event(db, task_id, "tests_blocked", "tests", {"reason": reason})
            db.commit()
            return {"status": "blocked", "reason": reason}
        call = ToolCall(task_id=task_id, tool_name="JavaSandbox.run", status="running", arguments={"build_tool": tool, "source_file": source_file})
        db.add(call)
        stage.status = "running"; stage.started_at = stage.started_at or now()
        task.current_stage = "tests"
        record_event(db, task_id, "tests_started", "tests", {"build_tool": tool})
        db.commit()
        started = now()
        try:
            report = run_java_tests(sandbox_url=settings.sandbox_url, task_id=task_id, worktree_path=task_id,
                                    build_tool=tool, source_file=source_file, timeout_seconds=600)
            db.add(TestResult(task_id=task_id, build_tool=tool, command=report["command"], exit_code=report["exit_code"],
                              passed=report["passed"], stdout=report["stdout"], stderr=report["stderr"],
                              started_at=started, finished_at=now()))
            stage.status = "succeeded" if report["passed"] else "failed"
            stage.detail = {"exit_code": report["exit_code"], "timed_out": report["timed_out"], "duration_seconds": report["duration_seconds"]}
            task.status = stage.status
            task.error_message = None if report["passed"] else "Java tests failed; inspect the test report."
            payload = dict(patch.source_payload or {})
            payload["validated"] = bool(report["passed"])
            patch.source_payload = payload
            call.status = "succeeded"; call.result = {"passed": report["passed"], "exit_code": report["exit_code"]}
            record_event(db, task_id, "tests_completed", "tests", {"passed": report["passed"], "exit_code": report["exit_code"]})
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            stage.status = "blocked"; stage.detail = {"reason": reason}
            task.status = "blocked"; task.error_message = reason
            call.status = "failed"; call.error = reason
            record_event(db, task_id, "tests_blocked", "tests", {"reason": reason})
        stage.finished_at = now(); task.finished_at = now(); call.finished_at = now()
        db.commit()
        return {"status": task.status}
