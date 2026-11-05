from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


def _git(args: list[str], *, timeout: int = 30, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", "core.longpaths=true", *args],
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorktreeError(str(exc)) from exc
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise WorktreeError(message or f"git exited with {result.returncode}")
    return result.stdout.strip()


def create_worktree(repository: str | Path, root: str | Path, task_id: str, revision: str = "HEAD") -> Path:
    repository_path = Path(repository).resolve()
    worktree_root = Path(root).resolve()
    if not repository_path.is_dir():
        raise WorktreeError("Repository path does not exist.")
    if not task_id or any(char not in "0123456789abcdef-" for char in task_id.lower()):
        raise WorktreeError("Task id contains unsupported characters.")
    try:
        top_level = Path(
            _git(["-C", str(repository_path), "rev-parse", "--show-toplevel"])
        ).resolve()
    except WorktreeError as exc:
        raise WorktreeError("Target repository must contain valid Git metadata.") from exc
    if top_level != repository_path:
        raise WorktreeError("Repository path must point to the Git working-tree root.")
    if _git(["-C", str(repository_path), "status", "--porcelain"]):
        raise WorktreeError(
            "Target repository has uncommitted changes. Stage a clean copy before creating a worktree."
        )
    worktree_root.mkdir(parents=True, exist_ok=True)
    destination = (worktree_root / task_id).resolve()
    if worktree_root not in destination.parents:
        raise WorktreeError("Worktree destination escapes the configured worktree root.")
    if destination.exists():
        raise WorktreeError("Worktree destination already exists.")
    _git(
        ["-C", str(repository_path), "worktree", "add", "--detach", str(destination), revision],
        timeout=60,
    )
    return destination


def apply_patch(worktree: str | Path, diff_text: str) -> None:
    path = Path(worktree).resolve()
    if not diff_text.strip():
        raise WorktreeError("Patch is empty.")
    _git(["-C", str(path), "apply", "--check", "--"], input_text=diff_text)
    _git(["-C", str(path), "apply", "--"], input_text=diff_text)


def diff(worktree: str | Path) -> str:
    path = Path(worktree).resolve()
    return _git(["-C", str(path), "diff", "--no-ext-diff", "--unified=3"])


def remove_worktree(repository: str | Path, worktree: str | Path, root: str | Path) -> None:
    repository_path = Path(repository).resolve()
    worktree_root = Path(root).resolve()
    worktree_path = Path(worktree).resolve()
    if worktree_root not in worktree_path.parents:
        raise WorktreeError("Refusing to remove a worktree outside the configured worktree root.")
    _git(
        ["-C", str(repository_path), "worktree", "remove", "--force", str(worktree_path)],
        timeout=60,
    )
