"""
Path priors and issue keyword extraction for code retrieval / GraphRAG ordering.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set


_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")
_GENERIC_CODE_IDENTIFIERS = {
    "add",
    "builder",
    "class",
    "config",
    "create",
    "default",
    "error",
    "exception",
    "factory",
    "false",
    "function",
    "get",
    "interface",
    "issue",
    "java",
    "json",
    "method",
    "null",
    "object",
    "parser",
    "private",
    "public",
    "read",
    "reader",
    "remove",
    "return",
    "set",
    "string",
    "test",
    "tests",
    "true",
    "type",
    "types",
    "value",
    "values",
    "write",
    "writer",
}


def path_priority_for_localization(rel_fname: str) -> float:
    """
    Higher = prefer for bug localization (subtracted from vector distance when reranking).
    Down-rank tests, Android compat suites, benchmarks, fuzz/JMH.

    Priority tiers (strengthened for src/main/java):
      - src/main/java/  core business logic  →  +6.0 (strongest signal)
      - src/main/       other main sources    →  +3.0
      - test/bench/demo paths                →  -6.0 (harder suppression)
    """
    s = (rel_fname or "").replace("\\", "/").lower()
    strong_negative = (
        "/test/",
        "src/test/",
        "androidtest",
        "/android/",
        "/benchmark/",
        "/jmh/",
        "jmh/",
        "/fuzz/",
        "/failing/",
        "/experimental/",
        "/samples/",
        "/sample/",
        "/demo/",
        "/it/",
        "integrationtest",
    )
    for pat in strong_negative:
        if pat in s:
            return -6.0
    if "src/main/java/" in s:
        return 6.0
    if s.startswith("src/") and "/main/" in s:
        return 3.0
    # Mild positive for src/ paths that aren't test
    if s.startswith("src/"):
        return 0.5
    return 0.0


def issue_keyword_tokens(issue_text: str, max_tokens: int = 48) -> Set[str]:
    """Lightweight tokens: CamelCase splits + alnum words, for neighbor boosting."""
    if not issue_text:
        return set()
    # Hints are excluded from lexical/symbol grounding even in the explicit
    # hint ablation, so a hint cannot directly steer deterministic retrieval.
    raw = re.split(
        r"(?im)^\s*(?:\[Dataset hints: ablation only\]|Hints:)\s*$",
        issue_text,
        maxsplit=1,
    )[0][:12000]
    # split CamelCase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)
    words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", s)
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "not",
        "can",
        "java",
        "null",
        "true",
        "false",
        "http",
        "https",
        "org",
        "com",
    }
    out: List[str] = []
    seen: Set[str] = set()
    for w in words:
        lw = w.lower()
        if lw in stop or len(lw) < 3:
            continue
        if lw not in seen:
            seen.add(lw)
            out.append(lw)
        if len(out) >= max_tokens:
            break
    return set(out)


def keyword_hits_in_text(text_lower: str, tokens: Set[str]) -> int:
    if not text_lower or not tokens:
        return 0
    return sum(1 for t in tokens if t and t in text_lower)


def issue_code_identifier_weights(issue_text: str) -> Dict[str, float]:
    """Extract high-precision code identifiers from pre-fix issue text.

    The extractor deliberately ignores free-form benchmark hints and assigns
    stronger weights to explicit ``.java`` names, stack frames, inline code,
    and call syntax.  Ordinary prose is used only when its token has a
    code-like shape (CamelCase, acronym, or embedded digit).
    """
    if not issue_text:
        return {}
    text = re.split(
        r"(?im)^\s*(?:\[Dataset hints: ablation only\]|Hints:)\s*$",
        issue_text,
        maxsplit=1,
    )[0][:12000]
    weighted: Dict[str, float] = {}

    def _add(raw: str, weight: float) -> None:
        token = str(raw or "").strip("`'\"")
        if not _IDENTIFIER_RE.fullmatch(token):
            return
        key = token.lower()
        if len(key) < 3 or key in _GENERIC_CODE_IDENTIFIERS:
            return
        weighted[key] = max(weighted.get(key, 0.0), float(weight))

    for match in re.finditer(r"\b([A-Za-z_$][A-Za-z0-9_$]+)\.java\b", text):
        _add(match.group(1), 18.0)
    for match in re.finditer(
        r"(?m)^\s*at\s+(?:[\w$]+\.)*([A-Za-z_$][A-Za-z0-9_$]*)\."
        r"([A-Za-z_$][A-Za-z0-9_$]*)",
        text,
    ):
        _add(match.group(1), 16.0)
        _add(match.group(2), 10.0)
    for match in re.finditer(r"`([^`\n]{2,100})`", text):
        for token in _IDENTIFIER_RE.findall(match.group(1)):
            _add(token, 11.0)
    for match in re.finditer(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(", text):
        _add(match.group(1), 8.0)
    for token in _IDENTIFIER_RE.findall(text):
        if (
            re.search(r"[a-z][A-Z]", token)
            or re.search(r"[A-Z].*[A-Z]", token)
            or any(character.isdigit() for character in token)
        ):
            _add(token, 7.0)
    return weighted


def rank_issue_entity_nodes(
    nodes: Mapping[str, Mapping[str, object]],
    issue_text: str,
    *,
    allowed_node_ids: Optional[Set[str]] = None,
    max_files: int = 24,
    max_nodes: int = 64,
    max_nodes_per_file: int = 3,
) -> List[str]:
    """Rank definition nodes grounded by explicit issue-code entities.

    Scores use repository-local inverse file frequency, so a distinctive
    method/class name carries more evidence than a ubiquitous lifecycle name.
    The result is a complementary retrieval channel; it never reads labels or
    treats a textual mention as proof of fault.
    """
    identifier_weights = issue_code_identifier_weights(issue_text)
    if not identifier_weights or max_files <= 0 or max_nodes <= 0:
        return []

    files_by_identifier: Dict[str, Set[str]] = defaultdict(set)
    matches_by_file: Dict[str, Set[str]] = defaultdict(set)
    matches_by_node: Dict[str, Set[str]] = defaultdict(set)
    nodes_by_file: Dict[str, List[str]] = defaultdict(list)
    all_files: Set[str] = set()

    for raw_node_id, data in nodes.items():
        node_id = str(raw_node_id)
        if allowed_node_ids is not None and node_id not in allowed_node_ids:
            continue
        if str(data.get("kind") or "").lower() != "def":
            continue
        path = str(data.get("rel_fname") or data.get("fname") or "").replace(
            "\\", "/"
        )
        if not path:
            continue
        all_files.add(path)
        nodes_by_file[path].append(node_id)
        basename = re.sub(
            r"\.java$", "", path.rsplit("/", 1)[-1], flags=re.IGNORECASE
        ).lower()
        symbol = str(data.get("name") or data.get("symbol_name") or "").lower()
        file_matches = {
            value for value in (basename, symbol) if value in identifier_weights
        }
        # A basename mention is file-level evidence, not evidence for every
        # method in that file. Only the node's own symbol grounds that node.
        node_matches = {symbol} if symbol in identifier_weights else set()
        for identifier in file_matches:
            files_by_identifier[identifier].add(path)
            matches_by_file[path].add(identifier)
        matches_by_node[node_id].update(node_matches)

    if not matches_by_file:
        return []
    file_count = max(1, len(all_files))

    def _identifier_score(identifier: str) -> float:
        document_frequency = max(1, len(files_by_identifier[identifier]))
        return identifier_weights[identifier] * math.log1p(
            file_count / document_frequency
        )

    file_scores: Dict[str, float] = {}
    for path, identifiers in matches_by_file.items():
        score = sum(_identifier_score(identifier) for identifier in identifiers)
        basename = re.sub(
            r"\.java$", "", path.rsplit("/", 1)[-1], flags=re.IGNORECASE
        ).lower()
        if basename in identifiers:
            score *= 1.25
        file_scores[path] = score

    ranked_files = sorted(file_scores, key=lambda path: (-file_scores[path], path))[
        :max_files
    ]
    ranked_nodes: List[str] = []
    for path in ranked_files:
        basename = re.sub(
            r"\.java$", "", path.rsplit("/", 1)[-1], flags=re.IGNORECASE
        ).lower()

        def _node_key(node_id: str) -> tuple[float, int, str]:
            identifiers = matches_by_node.get(node_id, set())
            score = sum(_identifier_score(identifier) for identifier in identifiers)
            if basename in identifiers:
                score *= 1.25
            is_class_anchor = int(basename not in identifiers)
            return (-score, is_class_anchor, node_id)

        file_nodes = [
            node_id for node_id in nodes_by_file[path] if matches_by_node.get(node_id)
        ]
        for node_id in sorted(file_nodes, key=_node_key)[:max_nodes_per_file]:
            if node_id not in ranked_nodes:
                ranked_nodes.append(node_id)
                if len(ranked_nodes) >= max_nodes:
                    return ranked_nodes
    return ranked_nodes
