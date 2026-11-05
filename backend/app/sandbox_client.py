from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any


class SandboxUnavailable(RuntimeError):
    pass


def run_java_tests(
    *,
    sandbox_url: str,
    task_id: str,
    worktree_path: str,
    build_tool: str,
    source_file: str = "",
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    if build_tool not in {"maven", "gradle"}:
        raise ValueError("Only Maven and Gradle test commands are supported.")
    payload = json.dumps(
        {
            "task_id": task_id,
            "worktree_path": worktree_path,
            "build_tool": build_tool,
            "source_file": source_file,
            "timeout_seconds": max(30, min(timeout_seconds, 900)),
        }
    ).encode("utf-8")
    request = Request(
        sandbox_url.rstrip("/") + "/run",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds + 15) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SandboxUnavailable(f"Java sandbox request failed: {exc}") from exc
