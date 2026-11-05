from __future__ import annotations

import re
from typing import List, Dict, Set

from langchain_core.tools import tool

import config
from agent.memory.unified_vectorstore import UnifiedVectorStore, get_shared_unified_vector_store


def _extract_paths_from_few_shot_block(block: str) -> Set[str]:
    """Extract file paths from a few-shot text block.

    Handles common formats produced by LocalizeAgent output:
    - ``候选文件: <path>``
    - ``File: <path>``
    - ``文件: <path>``
    - ``<path>.java`` mentions
    """
    paths: Set[str] = set()
    # Pattern 1: 候选文件: <path>
    for m in re.finditer(r"候选文件:\s*(\S+\.java)", block):
        paths.add(m.group(1))
    # Pattern 2: File: <path>
    for m in re.finditer(r"^File:\s*(\S+\.java)", block, re.MULTILINE):
        paths.add(m.group(1))
    # Pattern 3: 文件: <path>
    for m in re.finditer(r"文件:\s*(\S+\.java)", block):
        paths.add(m.group(1))
    # Pattern 4: any .java path mention (more lenient)
    for m in re.finditer(r"(src/\S+\.java)", block):
        paths.add(m.group(1))
    return paths


def _block_has_src_main_java(block: str) -> bool:
    """Check if a few-shot block references any src/main/java/ path."""
    paths = _extract_paths_from_few_shot_block(block)
    if not paths:
        # Fallback: simple substring check
        return "src/main/java/" in block
    return any("src/main/java/" in p for p in paths)


def filter_and_rank_few_shots(
    few_shots_text: str,
    *,
    min_src_main_java_ratio: float = 0.5,
) -> str:
    """Filter few-shot examples using source-scope information only.

    Rules (applied in order):
    1. **Hard filter**: Remove any example that does NOT reference at least one
       ``src/main/java/`` path — unless fewer than ``min_src_main_java_ratio``
       examples survive, in which case keep the best non-src/main/java ones.
    2. **Stable order**: Preserve retrieval order; no benchmark labels are used.
    3. **Re-number**: Surviving examples are re-numbered as Few-shot #1, #2, …

    Returns the filtered and re-ranked few-shot text (same format). If the result
    is empty, returns the original text unchanged (conservative fallback).
    """
    if not few_shots_text or not few_shots_text.strip():
        return few_shots_text

    # Split into individual few-shot blocks
    # Pattern: "Few-shot #N:" or "Few-shot #N\n"
    blocks = re.split(r"(?=Few-shot #\d+)", few_shots_text)
    if len(blocks) <= 1:
        # Try alternative format: numbered list starting with digits
        blocks = re.split(r"(?=^\d+\.)", few_shots_text, flags=re.MULTILINE)
    if len(blocks) <= 1:
        # Single block — just check src/main/java presence
        if "src/main/java/" not in few_shots_text:
            return few_shots_text  # don't filter a single block, let it through
        return few_shots_text

    # Separate into src/main/java blocks and others
    src_main_blocks: List[str] = []
    other_blocks: List[str] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if _block_has_src_main_java(block):
            src_main_blocks.append(block)
        else:
            other_blocks.append(block)

    # Determine how many to keep
    total = len(src_main_blocks) + len(other_blocks)
    if total == 0:
        return few_shots_text

    # If we have enough src/main/java blocks, drop the others
    min_needed = max(1, int(total * min_src_main_java_ratio))
    if len(src_main_blocks) >= min_needed:
        selected = src_main_blocks
    else:
        # Preserve retrieval order when supplementing with other blocks.
        needed_from_other = min_needed - len(src_main_blocks)
        selected = src_main_blocks + other_blocks[:needed_from_other]

    if not selected:
        return few_shots_text

    # Re-number and format
    out_blocks: List[str] = []
    for i, block in enumerate(selected, start=1):
        # Replace old "Few-shot #N" with new numbering
        renumbered = re.sub(r"Few-shot #\d+", f"Few-shot #{i}", block, count=1)
        out_blocks.append(renumbered)

    result = "\n\n".join(out_blocks)
    if not result.strip():
        return few_shots_text
    return result


def _get_store(
    repograph_dir: str,
    repo_key: str,
    vectorstore_path: str,
    embeddings_model: str,
    rebuild_vectorstore: bool,
) -> UnifiedVectorStore:
    store = get_shared_unified_vector_store(
        vectorstore_path,
        embeddings_model,
        rebuild=rebuild_vectorstore,
    )
    store.ensure_init(repograph_dir=repograph_dir, repo_key=repo_key)
    return store


@tool
def memory_few_shots(
    repograph_dir: str,
    repo_key: str,
    issue_text: str,
    k: int = 5,
    vectorstore_path: str = config.VECTORSTORE_PATH,
    embeddings_model: str = config.CODE_EMBEDDING_MODEL,
    rebuild_vectorstore: bool = False,
) -> str:
    """
    Retrieve long-term memory few-shot examples (issue_example docs).
    Returns a prompt-ready string.
    """
    store = _get_store(
        repograph_dir=repograph_dir,
        repo_key=repo_key,
        vectorstore_path=vectorstore_path,
        embeddings_model=embeddings_model,
        rebuild_vectorstore=rebuild_vectorstore,
    )
    docs = store.search_issue_examples(issue_text=issue_text, repo_key=repo_key, k=k)
    if not docs:
        return (
            "（暂无长期记忆中的相似 Issue 示例。请完全依据下方 GraphRAG 与 Issue 描述推断；"
            "优先主源码路径，避免仅凭目录名猜测。）"
        )

    blocks: List[str] = []
    for i, d in enumerate(docs, start=1):
        content = d.page_content
        blocks.append(f"Few-shot #{i}:\n{content}")
    return "\n\n".join(blocks)


@tool
def add_issue_example_to_memory(
    repograph_dir: str,
    repo_key: str,
    issue_text: str,
    localization_text: str,
    vectorstore_path: str = config.VECTORSTORE_PATH,
    embeddings_model: str = config.CODE_EMBEDDING_MODEL,
    rebuild_vectorstore: bool = False,
) -> str:
    """
    Persist a localization result as long-term memory (issue_example) into the same vector store.
    """
    store = _get_store(
        repograph_dir=repograph_dir,
        repo_key=repo_key,
        vectorstore_path=vectorstore_path,
        embeddings_model=embeddings_model,
        rebuild_vectorstore=rebuild_vectorstore,
    )
    store.add_issue_example(
        repo_key=repo_key,
        issue_text=issue_text,
        localization_text=localization_text,
    )
    return "ok"
