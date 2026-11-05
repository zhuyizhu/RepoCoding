from __future__ import annotations


def safe_truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    s = str(text)
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit]

