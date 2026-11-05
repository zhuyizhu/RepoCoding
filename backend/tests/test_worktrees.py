import subprocess
from pathlib import Path

import pytest

from app.worktrees import WorktreeError, apply_patch, create_worktree, diff, remove_worktree


def git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def init_repository(path: Path) -> None:
    path.mkdir(parents=True)
    git("init", cwd=path)
    git("config", "user.email", "demo@example.invalid", cwd=path)
    git("config", "user.name", "RepoCoding Test", cwd=path)
    (path / "src").mkdir()
    (path / "src" / "Hello.java").write_text("class Hello {}\n", encoding="utf-8")
    git("add", ".", cwd=path)
    git("commit", "-m", "initial", cwd=path)


def test_patch_is_applied_only_in_isolated_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    worktree_root = tmp_path / "worktrees"
    init_repository(repository)

    worktree = create_worktree(repository, worktree_root, "a" * 36)
    patch = (
        "--- a/src/Hello.java\n"
        "+++ b/src/Hello.java\n"
        "@@ -1 +1 @@\n"
        "-class Hello {}\n"
        "+class Hello { int value; }\n"
    )
    apply_patch(worktree, patch)

    assert "int value" in diff(worktree)
    assert (repository / "src" / "Hello.java").read_text(encoding="utf-8") == "class Hello {}\n"
    remove_worktree(repository, worktree, worktree_root)
    assert not worktree.exists()


def test_worktree_rejects_repository_with_uncommitted_changes(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    init_repository(repository)
    (repository / "untracked.txt").write_text("local change", encoding="utf-8")

    with pytest.raises(WorktreeError, match="uncommitted changes"):
        create_worktree(repository, tmp_path / "worktrees", "b" * 36)
