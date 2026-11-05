"""
Exclude test / benchmark sources from Java repograph and GraphRAG indexing.
"""
from __future__ import annotations


def is_test_or_benchmark_java_path(rel_path: str) -> bool:
    if not rel_path:
        return False
    # Prefixing a slash makes root-relative ``src/test/...`` paths match the
    # same rules as module-relative ``module/src/test/...`` paths.
    p = "/" + rel_path.replace("\\", "/").lower().lstrip("/")
    if "/src/test/java/" in p or "/src/test/resources/" in p:
        return True
    if "/src/test/groovy/" in p or "/src/test/kotlin/" in p:
        return True
    if "/test/java/" in p and "/src/" in p:
        return True
    if "/androidtest/" in p or "androidtest/" in p:
        return True
    if "/src/it/java/" in p or "/src/integrationtest/" in p:
        return True
    if "/integration-test/" in p or "/integration_test/" in p:
        return True
    if "/jmh/" in p or "/microbenchmark/" in p or "/micro-benchmark/" in p:
        return True
    return False


def is_localizable_java_source_path(rel_path: str) -> bool:
    """Return whether *rel_path* belongs to the Java source scope indexed by Repograph.

    Gold extraction and graph construction must use the same scope.  Otherwise a
    patch to documentation, build files, tests, or a newly absent source can be
    counted as a target even though the online localizer cannot retrieve it.
    """
    if not rel_path:
        return False
    normalized = rel_path.replace("\\", "/")
    return normalized.lower().endswith(".java") and not is_test_or_benchmark_java_path(normalized)
