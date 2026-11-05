"""
Git clone / fetch helpers for dataset repo cache (no heavy deps).
Used by construct_graph.py and dataset_repo_clone.py.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _git_clone_env() -> dict:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def _github_https_clone_url(org: str, repo: str, mirror_prefix: str = "") -> str:
    canonical = f"https://github.com/{org}/{repo}.git"
    p = (mirror_prefix or "").strip().rstrip("/")
    if not p:
        return canonical
    # 常见误写：前缀里已带 `.../https://github.com`，再拼 canonical 会变成双重 github.com
    gh_host = "https://github.com"
    while True:
        if p.endswith("/" + gh_host):
            p = p[: -len("/" + gh_host)].rstrip("/")
        elif p == gh_host:
            p = ""
            break
        else:
            break
    if not p:
        return canonical
    return f"{p}/{canonical}"


def _git_clone_http_opts() -> List[str]:
    return [
        "-c",
        "http.version=HTTP/1.1",
        "-c",
        "http.postBuffer=524288000",
        "-c",
        "http.lowSpeedLimit=0",
    ]


def _git_clone_no_checkout(urls: Union[str, List[str]], dest: str) -> None:
    url_list: List[str] = [urls] if isinstance(urls, str) else list(urls)
    url_list = [u.strip() for u in url_list if u and str(u).strip()]
    if not url_list:
        raise ValueError("git clone: no URLs provided")

    dest_path = Path(dest)
    env = _git_clone_env()
    http = _git_clone_http_opts()
    all_logs: List[str] = []

    def clean_dest() -> None:
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)

    def run_clone_one(url: str) -> subprocess.CompletedProcess[str]:
        cmd = ["git", *http, "clone", "--no-checkout", url, dest]
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    delays_sec = (0, 8, 24, 60)
    for url_idx, url in enumerate(url_list, start=1):
        logs: List[str] = []
        for attempt, delay in enumerate(delays_sec, start=1):
            if delay:
                time.sleep(delay)
            clean_dest()
            proc = run_clone_one(url)
            if proc.returncode == 0:
                return
            logs.append(
                f"[url {url_idx}/{len(url_list)} attempt {attempt}/{len(delays_sec)}] rc={proc.returncode}\n{proc.stdout}"
            )
        all_logs.append(f"=== Clone URL failed: {url} ===\n" + "\n---\n".join(logs))

    raise RuntimeError(
        "git clone failed for all candidate URLs (mirror may be down — use direct GitHub or another mirror).\n"
        f"  tried={url_list}\n  dest={dest}\n"
        + "\n".join(all_logs)
    )


def safe_version_dirname(version_id: str) -> str:
    """Filesystem-safe directory name for a dataset version_id (e.g. main-12b40c7)."""
    s = re.sub(r"[^\w.\-]+", "_", (version_id or "").strip())
    s = s.strip("._-")[:180]
    return s if s else "unknown"


def _git_rev_parse(main_clone: Path, rev: str) -> Optional[str]:
    p = subprocess.run(
        ["git", "-C", str(main_clone), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if p.returncode != 0:
        return None
    return (p.stdout or "").strip()


def _git_worktree_head(wt: Path) -> Optional[str]:
    p = subprocess.run(
        ["git", "-C", str(wt), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if p.returncode != 0:
        return None
    return (p.stdout or "").strip()


def git_fetch_commit(main_clone: Path, sha: str) -> None:
    """Ensure commit object is available locally (same http tuning as clone)."""
    env = _git_clone_env()
    http = _git_clone_http_opts()
    subprocess.run(
        ["git", *http, "-C", str(main_clone), "fetch", "--no-tags", "origin", sha],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def ensure_git_worktree(main_clone: Path, worktree_path: Path, sha: str) -> None:
    """
    Create or refresh a detached worktree at `worktree_path` for commit `sha`.
    `main_clone` is the primary clone (contains .git). Layout:
      <repo_work_root>/<repo>/   <- main_clone
      <repo_work_root>/versions/<safe_version_id>/  <- worktree_path
    """
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    full_sha = _git_rev_parse(main_clone, sha)
    if not full_sha:
        git_fetch_commit(main_clone, sha)
        full_sha = _git_rev_parse(main_clone, sha)
    if not full_sha:
        raise RuntimeError(f"git: cannot resolve commit {sha!r} in {main_clone}")

    if worktree_path.exists():
        head = _git_worktree_head(worktree_path)
        if head == full_sha:
            return
        if head is not None:
            subprocess.run(
                ["git", "-C", str(main_clone), "worktree", "remove", "--force", str(worktree_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        shutil.rmtree(worktree_path, ignore_errors=True)

    # 若目录曾被手动删掉，Git 仍可能在 .git/worktrees 里登记该路径，导致 worktree add 报 already exists。
    if not worktree_path.exists():
        subprocess.run(
            ["git", "-C", str(main_clone), "worktree", "prune"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=_git_clone_env(),
        )

    env = _git_clone_env()
    http = _git_clone_http_opts()
    add = subprocess.run(
        ["git", *http, "-C", str(main_clone), "worktree", "add", "--detach", str(worktree_path), full_sha],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if add.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {worktree_path}\n{add.stdout}")


def write_repo_versions_manifest(
    repo_work_root: Path,
    org: str,
    repo: str,
    main_clone_rel: str,
    versions: Dict[str, Dict[str, Any]],
    layout: str,
) -> None:
    doc = {
        "layout": layout,
        "org": org,
        "repo": repo,
        "main_clone_rel": main_clone_rel,
        "versions": versions,
    }
    p = repo_work_root / "repo_versions_manifest.json"
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
