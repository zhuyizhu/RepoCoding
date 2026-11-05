from __future__ import annotations

import json
import os
import signal
import threading
import re
import subprocess
import time
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("WORKTREE_ROOT", "/workspace/worktrees")).resolve()
OUTPUT_LIMIT = 24000


def collect_process(process, timeout):
    tails = ["", ""]
    def drain(stream,index):
        while True:
            chunk = stream.read(4096)
            if not chunk:break
            tails[index]=(tails[index]+chunk)[-OUTPUT_LIMIT:]
        stream.close()
    threads=[threading.Thread(target=drain,args=(stream,i),daemon=True) for i,stream in enumerate((process.stdout,process.stderr))]
    for thread in threads:thread.start()
    timed_out=False
    try:process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
    finally:
        # Terminate lingering descendants as well as a timed-out build.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()
    for thread in threads:thread.join(timeout=3)
    return tails[0],tails[1],timed_out



def response(status: int, body: dict[str, Any]) -> tuple[int, bytes]:
    return status, json.dumps(body, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "RepoCodingSandbox/0.1"

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send(*response(404, {"error": "not found"}))
            return
        status, body = response(200, {"status": "ready"})
        self._send(status, body)

    def do_POST(self) -> None:
        if self.path != "/run":
            self._send(*response(404, {"error": "not found"}))
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 2 or size > 4096:
                raise ValueError("Invalid request size.")
            request = json.loads(self.rfile.read(size))
            task_id = str(request.get("task_id", ""))
            relative = str(request.get("worktree_path", "")).replace("\\", "/")
            build_tool = str(request.get("build_tool", ""))
            source_file = str(request.get("source_file", ""))
            timeout = max(30, min(int(request.get("timeout_seconds", 900)), 900))
            if build_tool not in {"maven", "gradle"}:
                raise ValueError("build_tool must be maven or gradle.")
            if not re.fullmatch(r"[a-f0-9-]{36}",task_id) or relative != task_id:
                raise ValueError("Worktree path must stay under the mounted worktree root.")
            worktree = (ROOT / relative).resolve()
            if ROOT not in worktree.parents or not worktree.is_dir():
                raise ValueError("Worktree directory was not found under the mounted root.")
            source_path = Path(source_file)
            if (not source_file or source_path.is_absolute() or ".." in source_path.parts
                    or source_path.suffix != ".java" or not (worktree / source_path).is_file()):
                raise ValueError("source_file must name an existing Java file in this worktree.")

            with tempfile.TemporaryDirectory(prefix="java-test-", dir="/tmp") as scratch:
                working = Path(scratch) / "repo"
                shutil.copytree(worktree, working, symlinks=True, ignore=shutil.ignore_patterns(".git"))
                command = self._command(working, build_tool, source_path)
                body = self._execute(command, working, timeout, task_id, build_tool)
            self._send(200, json.dumps(body, ensure_ascii=False).encode("utf-8"))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(*response(400, {"error": str(exc)}))
        except Exception as exc:
            self._send(*response(500, {"error": type(exc).__name__ + ": " + str(exc)}))

    @staticmethod
    def _execute(command: list[str], working: Path, timeout: int, task_id: str, build_tool: str) -> dict[str, Any]:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "JAVA_HOME": os.environ.get("JAVA_HOME", ""),
                "HOME": "/tmp/runner-home",
                "TMPDIR": "/tmp",
                "MAVEN_CONFIG": "/cache/m2",
                "GRADLE_USER_HOME": "/cache/gradle",
                "CI": "true",
            }
            Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=working,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            stdout, stderr, timed_out = collect_process(process, timeout)
            elapsed = round(time.monotonic() - started, 3)
            return {
                "task_id": task_id,
                "build_tool": build_tool,
                "command": command,
                "exit_code": process.returncode,
                "passed": process.returncode == 0 and not timed_out,
                "timed_out": timed_out,
                "duration_seconds": elapsed,
                "stdout": stdout[-OUTPUT_LIMIT:],
                "stderr": stderr[-OUTPUT_LIMIT:],
            }
    @staticmethod
    def _command(worktree: Path, tool: str, source_file: Path) -> list[str]:
        if tool == "maven":
            command = ["mvn", "-o", "-B", "-Dmaven.repo.local=/cache/m2"]
            parts = source_file.parts
            for index in range(len(parts) - 2):
                if parts[index:index + 3] == ("src", "main", "java"):
                    module = Path(*parts[:index]) if index else Path(".")
                    test_file = module / "src" / "test" / "java" / Path(*parts[index + 3:-1]) / (source_file.stem + "Test.java")
                    if (worktree / test_file).is_file() and (worktree / module / "pom.xml").is_file():
                        if str(module) != ".":
                            command += ["-pl", module.as_posix(), "-am"]
                        command += ["-Dtest=" + source_file.stem + "Test", "-Dsurefire.failIfNoSpecifiedTests=false"]
                    break
            return command + ["test"]
        return ["gradle", "--offline", "test", "--no-daemon"]

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
