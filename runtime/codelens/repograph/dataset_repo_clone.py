"""
Collect unique (org, repo) and all dataset versions (base.ref + base.sha) from
Multi-SWE-bench *_dataset.jsonl, then:

1. Clone each repo to ``<cache_root>/<org>__<repo>/<repo>/`` (same as construct_graph).
2. Optionally create detached git worktrees under ``<cache_root>/<org>__<repo>/versions/<safe_version_id>/``
   for every version, and write ``repo_versions_manifest.json`` (same fields/layout as construct_graph
   with ``use_version_worktrees``).

``construct_graph`` / localization resolve a version via ``version_id`` = ``{ref}-{sha[:7]}`` and
repograph dirs ``<output_root>/<repo>/<version_id>/``; the manifest maps each ``version_id`` to
``worktree_rel`` so tooling can open the matching checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Allow `python repograph/dataset_repo_clone.py` and imports from project root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from git_cache_utils import (  # noqa: E402
    _git_clone_env,
    _git_clone_http_opts,
    _github_https_clone_url,
    _git_clone_no_checkout,
    ensure_git_worktree,
    safe_version_dirname,
    write_repo_versions_manifest,
)


def collect_unique_repos(dataset_jsonl_dir: Path) -> List[Tuple[str, str]]:
    seen: Set[str] = set()
    out: List[Tuple[str, str]] = []
    for df in sorted(dataset_jsonl_dir.glob("*_dataset.jsonl")):
        with open(df, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                org = rec.get("org", "")
                repo = rec.get("repo", "")
                if not org or not repo:
                    continue
                key = f"{org}/{repo}"
                if key not in seen:
                    seen.add(key)
                    out.append((org, repo))
    return out


def collect_versions_by_repo(dataset_jsonl_dir: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return ``org/repo`` -> sorted unique rows ``(full_sha, ref, version_id)``."""
    acc: Dict[str, Dict[str, Tuple[str, str, str]]] = {}
    for df in sorted(dataset_jsonl_dir.glob("*_dataset.jsonl")):
        with open(df, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                org = rec.get("org", "")
                repo = rec.get("repo", "")
                base = rec.get("base") or {}
                sha = (base.get("sha") or "").strip()
                ref = (base.get("ref") or "").strip()
                if not org or not repo or not sha or not ref:
                    continue
                key = f"{org}/{repo}"
                version_id = f"{ref}-{sha[:7]}"
                acc.setdefault(key, {})[version_id] = (sha, ref, version_id)
    out: Dict[str, List[Tuple[str, str, str]]] = {}
    for key, vmap in acc.items():
        out[key] = sorted(vmap.values(), key=lambda t: (t[1], t[0]))
    return out


def _is_git_repo(path: Path) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--git-dir"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _fetch_origin(repo_dir: Path) -> Tuple[bool, str]:
    http = _git_clone_http_opts()
    proc = subprocess.run(
        ["git", *http, "-C", str(repo_dir), "fetch", "--no-tags", "--prune", "origin"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_git_clone_env(),
    )
    return proc.returncode == 0, proc.stdout or ""


def clone_one_repo(
    org: str,
    repo: str,
    cache_root: Path,
    github_clone_prefix: str,
    clone_fallback_direct: bool,
    update_existing: bool,
) -> Tuple[str, str]:
    """Returns (status_line, detail). status_line starts with OK / SKIP / FAIL."""
    work = cache_root / f"{org}__{repo}"
    clone_dir = work / repo
    work.mkdir(parents=True, exist_ok=True)

    canonical = f"https://github.com/{org}/{repo}.git"
    mirror_url = _github_https_clone_url(org, repo, github_clone_prefix)
    clone_urls: List[str] = []
    if github_clone_prefix and mirror_url != canonical:
        clone_urls.append(mirror_url)
    if clone_fallback_direct or not github_clone_prefix:
        if canonical not in clone_urls:
            clone_urls.append(canonical)
    elif not clone_urls:
        clone_urls.append(canonical)

    label = f"{org}/{repo}"

    if clone_dir.exists() and _is_git_repo(clone_dir):
        if update_existing:
            ok, out = _fetch_origin(clone_dir)
            if ok:
                return f"OK {label} (fetch)", ""
            return f"FAIL {label} (fetch)", out[-2000:]
        return f"SKIP {label} (exists)", ""

    try:
        if clone_dir.exists():
            import shutil

            shutil.rmtree(clone_dir, ignore_errors=True)
        _git_clone_no_checkout(clone_urls, str(clone_dir))
        return f"OK {label} (clone)", ""
    except Exception as e:
        return f"FAIL {label}", str(e)[:2000]


def materialize_version_worktrees(
    org: str,
    repo: str,
    cache_root: Path,
    version_rows: List[Tuple[str, str, str]],
) -> Tuple[str, str]:
    """
    Fetch latest refs, create worktrees under ``versions/<safe_version_id>/``, write manifest.
    ``version_rows``: ``(full_sha, ref, version_id)``.
    """
    work = cache_root / f"{org}__{repo}"
    clone_dir = work / repo
    label = f"{org}/{repo}"

    if not clone_dir.is_dir() or not _is_git_repo(clone_dir):
        return f"SKIP {label} versions (no git clone)", ""

    _ok_fetch, fetch_out = _fetch_origin(clone_dir)

    manifest_versions: Dict[str, Dict[str, Any]] = {}
    for sha, ref, version_id in version_rows:
        safe_v = safe_version_dirname(version_id)
        wt_path = work / "versions" / safe_v
        try:
            ensure_git_worktree(clone_dir, wt_path, sha)
        except Exception as e:
            detail = (fetch_out or "")[-1500:] + "\n" + str(e)[:2000]
            return f"FAIL {label} worktree {version_id}", detail[-3500:]

        manifest_versions[version_id] = {
            "sha": sha,
            "ref": ref,
            "worktree_rel": f"versions/{safe_v}",
            "repograph_version_dir": version_id,
        }

    write_repo_versions_manifest(
        work,
        org,
        repo,
        main_clone_rel=repo,
        versions=manifest_versions,
        layout="git_worktrees",
    )
    return f"OK {label} ({len(version_rows)} worktrees + manifest)", ""


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clone all unique GitHub repos from multi-swe-bench-java dataset into a cache tree "
        "compatible with construct_graph.py (--repo-cache-root); optionally materialize per-version "
        "git worktrees and repo_versions_manifest.json."
    )
    parser.add_argument(
        "--dataset-jsonl-dir",
        type=str,
        required=True,
        help="Directory containing *_dataset.jsonl (e.g. datasets/multi-swe-bench-java).",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        required=True,
        help="Target cache root, same as construct_graph --repo-cache-root (e.g. /data/jloc_repo_cache).",
    )
    parser.add_argument(
        "--github-clone-prefix",
        type=str,
        default="",
        help="Mirror prefix; if empty, uses env GITHUB_CLONE_PREFIX.",
    )
    parser.add_argument(
        "--no-github-clone-fallback",
        action="store_true",
        help="Do not fall back to direct github.com when mirror fails.",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="For repos already cloned, run `git fetch` instead of skipping.",
    )
    parser.add_argument(
        "--no-version-worktrees",
        action="store_true",
        help="Only clone/fetch main repo; do not create versions/* worktrees or manifest.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel clone/fetch workers (default 1 = sequential; use 2–3 on stable networks).",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_jsonl_dir)
    if not dataset_dir.is_dir():
        print(f"ERROR: not a directory: {dataset_dir}", file=sys.stderr)
        return 1

    cache_root = Path(args.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    gh_prefix = (args.github_clone_prefix or "").strip().rstrip("/")
    if not gh_prefix:
        gh_prefix = (os.environ.get("GITHUB_CLONE_PREFIX") or "").strip().rstrip("/")

    version_worktrees = not args.no_version_worktrees
    versions_by_repo = collect_versions_by_repo(dataset_dir)
    repos = collect_unique_repos(dataset_dir)
    if not repos:
        print("No (org, repo) found in dataset jsonl files.", file=sys.stderr)
        return 1

    print(f"Cache root: {cache_root}")
    print(f"Unique repos: {len(repos)}")
    if version_worktrees:
        nver = sum(len(versions_by_repo.get(f'{o}/{r}', [])) for o, r in repos)
        print(f"Version worktrees: enabled (unique versions across dataset: {nver})")
    else:
        print("Version worktrees: disabled (--no-version-worktrees)")
    print(f"Set REPO_CACHE_ROOT={cache_root} or pass --repo-cache-root when running construct_graph.py\n")

    clone_fallback = not args.no_github_clone_fallback
    jobs = max(1, int(args.jobs))

    results: List[Tuple[str, str]] = []
    if jobs == 1:
        for org, repo in repos:
            r, d = clone_one_repo(
                org, repo, cache_root, gh_prefix, clone_fallback, args.update_existing
            )
            results.append((r, d))
            print(r)
            if d:
                print(d[:1500])
            time.sleep(0)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {
                ex.submit(
                    clone_one_repo,
                    org,
                    repo,
                    cache_root,
                    gh_prefix,
                    clone_fallback,
                    args.update_existing,
                ): (org, repo)
                for org, repo in repos
            }
            for fut in as_completed(futs):
                org, repo = futs[fut]
                try:
                    r, d = fut.result()
                except Exception as e:
                    r, d = f"FAIL {org}/{repo}", str(e)
                results.append((r, d))
                print(r)
                if d:
                    print(d[:1500])

    ok = sum(1 for r, _ in results if r.startswith("OK"))
    skip = sum(1 for r, _ in results if r.startswith("SKIP"))
    fail = sum(1 for r, _ in results if r.startswith("FAIL"))
    print(f"\nClone summary: OK={ok} SKIP={skip} FAIL={fail}")

    exit_code = 0 if fail == 0 else 2

    if version_worktrees:
        wt_results: List[Tuple[str, str]] = []
        v_jobs = jobs
        keys_sorted = sorted(versions_by_repo.keys())
        if v_jobs == 1:
            for key in keys_sorted:
                if "/" not in key:
                    continue
                org, repo = key.split("/", 1)
                rows = versions_by_repo[key]
                wr, wd = materialize_version_worktrees(org, repo, cache_root, rows)
                wt_results.append((wr, wd))
                print(wr)
                if wd:
                    print(wd[:1500])
        else:
            with ThreadPoolExecutor(max_workers=v_jobs) as ex:
                futs = {}
                for key in keys_sorted:
                    if "/" not in key:
                        continue
                    org, repo = key.split("/", 1)
                    rows = versions_by_repo[key]
                    futs[
                        ex.submit(materialize_version_worktrees, org, repo, cache_root, rows)
                    ] = key
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        wr, wd = fut.result()
                    except Exception as e:
                        wr, wd = f"FAIL {key} versions", str(e)
                    wt_results.append((wr, wd))
                    print(wr)
                    if wd:
                        print(wd[:1500])

        w_ok = sum(1 for r, _ in wt_results if r.startswith("OK"))
        w_skip = sum(1 for r, _ in wt_results if r.startswith("SKIP"))
        w_fail = sum(1 for r, _ in wt_results if r.startswith("FAIL"))
        print(f"\nWorktree/manifest summary: OK={w_ok} SKIP={w_skip} FAIL={w_fail}")
        if w_fail:
            exit_code = 2

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
