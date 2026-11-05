from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.orm.session import sessionmaker

from app.config import settings
from app.database import Base, SessionLocal, engine, get_db
from app.models import (
    CandidateHypothesis,
    OutboxMessage,
    Task,
    TaskEvent,
    TaskStage,
    ToolCall,
)
from app.schemas import (
    EventOut,
    HealthOut,
    TaskCreate,
    LocalDemoCreate,
    TaskCreated,
    TaskDetail,
    TaskSummary,
)
from app.events import record_event
from app.engine.evistate import handoff
from app.engine.persist_evistate import persist_snapshot
from app.engine.repair import run_quick_repair
from app.celery_app import celery_app

app = FastAPI(title="RepoCoding Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Last-Event-ID"],
)

if settings.auto_create_schema:
    Base.metadata.create_all(bind=engine)


def get_stream_session_factory() -> sessionmaker:
    return SessionLocal


@app.get("/health", response_model=HealthOut)
def health(db: Session = Depends(get_db)) -> HealthOut:
    db.execute(text("SELECT 1"))
    return HealthOut(
        status="ok",
        integrations={
            "api": "ready",
            "database": db.get_bind().dialect.name + " ready",
            "redis": "configured for Java validation" if settings.repair_enabled else "configured" if settings.full_pipeline_enabled else "not used",
            "codelens": "configured for research worker" if settings.full_pipeline_enabled else "disabled in minimal deployment",
            "evistate": "real core adapter configured; readiness checked by worker",
            "repair": "enabled" if settings.repair_enabled else "disabled pending Qwen/Java readiness",
            "docker": "runtime required for isolated Java tests",
        },
    )


@app.post("/api/tasks", response_model=TaskCreated, status_code=202)
def create_task(request: TaskCreate, db: Session = Depends(get_db)) -> TaskCreated:
    if not settings.full_pipeline_enabled:
        raise HTTPException(503, "Full CodeLens pipeline is not enabled in this deployment.")
    task = Task(
        issue_title=request.issue_title,
        issue_text=request.issue_text,
        repo_key=request.repo_key,
        repository_path=request.repository_path,
        repograph_path=request.repograph_path,
        status="queued",
        current_stage="queued",
    )
    db.add(task)
    db.flush()
    event = record_event(
        db,
        task.id,
        "task_created",
        "queued",
        {"message": "Task accepted; queued for the research worker."},
    )
    db.add(OutboxMessage(task_id=task.id, topic="run_localization", payload={}))
    db.commit()
    db.refresh(task)
    return TaskCreated(task=task, event_id=event.id)


@app.post("/api/demo", response_model=TaskCreated, status_code=201)
def inspect_file_without_model(request: LocalDemoCreate, db: Session = Depends(get_db)) -> TaskCreated:
    """User-selected file -> EviState; optional vLLM patch in a Git worktree."""
    root = Path(settings.repository_root).resolve()
    repository = (root / request.repository_path).resolve()
    graph = (Path(settings.repograph_root).resolve() / request.repograph_path).resolve()
    if root not in repository.parents or not repository.is_dir():
        raise HTTPException(422, "Repository directory is unavailable.")
    if not all((graph / name).is_file() for name in ("graph.pkl", "tags.json")):
        raise HTTPException(422, "Repograph directory is incomplete.")
    task = Task(issue_title=request.issue_title, issue_text=request.issue_text,
        repo_key=request.repo_key, repository_path=request.repository_path,
        repograph_path=request.repograph_path, status="running",current_stage="evistate")
    db.add(task);db.flush()
    first=record_event(db,task.id,"task_created","queued",{"mode":"manual_file_inspection","file_path":request.file_path})
    db.add(TaskStage(task_id=task.id,name="localization",status="skipped",detail={"reason":"File supplied by the user; CodeLens model was not called."},finished_at=datetime.now(timezone.utc)))
    db.add(TaskStage(task_id=task.id,name="evistate",status="running",started_at=datetime.now(timezone.utc)))
    call=ToolCall(task_id=task.id,tool_name="EviState.StateTracker+Barrier",status="running",arguments={"source":"manual_file_input","file_path":request.file_path})
    db.add(call);db.commit()
    try:
        result=handoff(candidates=[{"file_path":request.file_path,"target":request.file_path,"target_type":"file","confidence":0.0,"reason":"User-selected path; no model inference."}],
            task_id=task.id,repo_key=task.repo_key,repository=repository,
            evistate_home=settings.evistate_home,artifact_root=settings.artifact_root,source="manual_file_input")
        persist_snapshot(db,task,result)
        call.status="succeeded"
        call.result={"source_hash":result["snapshot"]["source_hash"],"selected_paths":result["paths"]}
        call.finished_at=datetime.now(timezone.utc)
        stage=db.query(TaskStage).filter_by(task_id=task.id,name="evistate").one()
        stage.status="succeeded";stage.detail={"source_hash":result["snapshot"]["source_hash"]};stage.finished_at=datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        task=db.get(Task,task.id)
        task.status="blocked";task.error_message=str(exc);task.finished_at=datetime.now(timezone.utc)
        stage=db.query(TaskStage).filter_by(task_id=task.id,name="evistate").one()
        stage.status="blocked";stage.detail={"reason":str(exc)};stage.finished_at=datetime.now(timezone.utc)
        call=db.get(ToolCall,call.id)
        call.status="blocked";call.error=str(exc);call.finished_at=datetime.now(timezone.utc)
        record_event(db,task.id,"stage_blocked","evistate",{"reason":str(exc)})
        db.commit()
        db.refresh(task)
        return TaskCreated(task=task,event_id=first.id)
    if request.repair_with_model:
        task.current_stage = "repair"
        repair_stage = TaskStage(task_id=task.id,name="repair",status="running",started_at=datetime.now(timezone.utc))
        db.add(repair_stage)
        record_event(db,task.id,"stage_started","repair",{"message":"Calling Qwen 7B for an isolated Java patch."})
        db.commit()
        try:
            patch = run_quick_repair(db,task,request.file_path,settings,result['code_version'])
            repair_stage.status = "succeeded"
            repair_stage.detail = patch
            repair_stage.finished_at = datetime.now(timezone.utc)
            task.status = "running"
            task.current_stage = "tests"
            db.add(TaskStage(task_id=task.id,name="tests",status="pending",detail={"reason":"Waiting for isolated Java validation."}))
            record_event(db,task.id,"patch_generated","repair",{"message":"Qwen patch generated; Java validation queued."})
            db.commit()
            try:
                celery_app.send_task("app.validation_worker.validate_patch", args=[task.id, request.file_path], queue="validation")
            except Exception as exc:
                stage = db.query(TaskStage).filter_by(task_id=task.id,name="tests").one()
                stage.status = "blocked"; stage.detail = {"reason": f"Could not queue Java validation: {exc}"}; stage.finished_at = datetime.now(timezone.utc)
                task.status = "blocked"; task.error_message = stage.detail["reason"]; task.finished_at = datetime.now(timezone.utc)
                record_event(db,task.id,"tests_blocked","tests",{"reason": task.error_message})
        except Exception as exc:
            repair_stage.status = "failed"
            repair_stage.detail = {"reason":str(exc)}
            task.status = "failed"
            task.error_message = str(exc)
            record_event(db,task.id,"task_failed","repair",{"reason":str(exc)})
            db.add(TaskStage(task_id=task.id,name="tests",status="skipped",detail={"reason":"Patch generation failed."},finished_at=datetime.now(timezone.utc)))
            task.finished_at = datetime.now(timezone.utc)
        repair_stage.finished_at = datetime.now(timezone.utc)
    else:
        db.add(TaskStage(task_id=task.id,name="repair",status="skipped",detail={"reason":"Model repair was not requested."},finished_at=datetime.now(timezone.utc)))
        task.status = "succeeded"
        record_event(db,task.id,"task_succeeded","evistate",{"message":"Real source file and EviState evidence inspected; CodeLens and repair not run."})
        db.add(TaskStage(task_id=task.id,name="tests",status="skipped",detail={"reason":"Model repair was not requested."},finished_at=datetime.now(timezone.utc)))
        task.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(task)
    return TaskCreated(task=task,event_id=first.id)


@app.get("/api/tasks", response_model=list[TaskSummary])
def list_tasks(
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[Task]:
    return list(db.scalars(select(Task).order_by(Task.created_at.desc()).limit(limit)).all())


@app.get("/api/tasks/{task_id}", response_model=TaskDetail)
def get_task(task_id: str, db: Session = Depends(get_db)) -> Task:
    task = db.scalar(
        select(Task)
        .where(Task.id == task_id)
        .options(
            selectinload(Task.stages),
            selectinload(Task.candidates).selectinload(CandidateHypothesis.bindings),
            selectinload(Task.evidence),
            selectinload(Task.patches),
            selectinload(Task.test_results),
            selectinload(Task.tool_calls),
        )
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.get("/api/tasks/{task_id}/events", response_model=list[EventOut])
def list_events(
    task_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> list[TaskEvent]:
    if db.get(Task, task_id) is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return list(
        db.scalars(
            select(TaskEvent)
            .where(TaskEvent.task_id == task_id, TaskEvent.id > after)
            .order_by(TaskEvent.id)
            .limit(limit)
        ).all()
    )


@app.get("/api/tasks/{task_id}/stream")
def stream_events(
    task_id: str,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    session_factory: sessionmaker = Depends(get_stream_session_factory),
) -> Response:
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError:
        cursor = after
    with session_factory() as db:
        if db.get(Task, task_id) is None:
            raise HTTPException(status_code=404, detail="Task not found.")

    def generate():
        current = cursor
        idle_ticks = 0
        while True:
            with session_factory() as stream_db:
                rows = list(
                    stream_db.scalars(
                        select(TaskEvent)
                        .where(TaskEvent.task_id == task_id, TaskEvent.id > current)
                        .order_by(TaskEvent.id)
                        .limit(500)
                    ).all()
                )
                task = stream_db.get(Task, task_id)
                status = task.status if task else "missing"
            for item in rows:
                current = item.id
                body = json.dumps(
                    {
                        "id": item.id,
                        "event_type": item.event_type,
                        "stage": item.stage,
                        "payload": item.payload,
                        "created_at": item.created_at.isoformat(),
                    },
                    ensure_ascii=False,
                    default=str,
                )
                yield f"id: {item.id}\nevent: task_event\ndata: {body}\n\n"
            if rows:
                idle_ticks = 0
            else:
                idle_ticks += 1
                if status in {"blocked", "succeeded", "failed", "cancelled", "missing"}:
                    yield f"event: stream_end\ndata: {json.dumps({'status': status})}\n\n"
                    break
                if idle_ticks % 20 == 0:
                    yield ": keep-alive\n\n"
                time.sleep(settings.event_poll_seconds)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
