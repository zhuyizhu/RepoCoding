"""Resolve CodeLens module-relative file paths against a real Git checkout."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any


def resolve_candidate_paths(
    candidates: list[dict[str, Any]], repository: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    root = repository.resolve()
    resolved: list[dict[str, Any]] = []
    mappings: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        native = str(item.get("file_path") or "").replace("\\", "/")
        relative = PurePosixPath(native)
        if (
            not native.endswith(".java")
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
        ):
            resolved.append(item)
            continue
        exact = (root / native).resolve()
        if exact.is_file() and root in exact.parents:
            resolved.append(item)
            continue
        suffix = "/" + native
        matches = []
        for path in root.rglob(relative.name):
            actual = path.resolve()
            if (
                actual.is_file()
                and root in actual.parents
                and not path.is_symlink()
                and ("/" + path.relative_to(root).as_posix()).endswith(suffix)
            ):
                matches.append(path.relative_to(root).as_posix())
                if len(matches) > 1:
                    break
        if len(matches) == 1:
            item["native_file_path"] = native
            item["file_path"] = matches[0]
            if item.get("target") == native:
                item["target"] = matches[0]
            item["path_resolution"] = "unique_repository_suffix"
            mappings.append({"native_file_path": native, "resolved_file_path": matches[0]})
        resolved.append(item)
    return resolved, mappings
