from __future__ import annotations

import importlib
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class IntegrationUnavailable(RuntimeError):
    pass


class LocalizationRunError(RuntimeError):
    pass


@dataclass
class LocalizationOutput:
    candidates: list[dict[str, Any]]
    policy_trajectory: dict[str, Any] | None
    retrieval_summary: list[dict[str, Any]]
    raw_output: str


def _inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise IntegrationUnavailable(f"Configured path escapes its mounted root: {relative}")
    if not resolved.exists():
        raise IntegrationUnavailable(f"Configured input does not exist: {relative}")
    return resolved


def vectorstore_path_for(vectorstore_root: str, repo_key: str, graph_dir: Path, embedding_model: str) -> Path:
    index_key = hashlib.sha256((str(graph_dir) + "\0" + embedding_model).encode()).hexdigest()[:12]
    return Path(vectorstore_root) / f"{repo_key}-{index_key}.faiss"


def _check_embedding_device() -> None:
    """Fail before indexing when the GPU deployment cannot use CUDA."""
    if os.getenv("CODELENS_REQUIRE_CUDA", "").lower() not in {"1", "true", "yes"}:
        return
    if not os.getenv("CODE_EMBEDDING_DEVICE", "").lower().startswith("cuda"):
        raise IntegrationUnavailable("CODELENS_REQUIRE_CUDA requires CODE_EMBEDDING_DEVICE=cuda.")
    try:
        import torch

        available = torch.cuda.is_available()
    except (ImportError, RuntimeError) as exc:
        raise IntegrationUnavailable(f"CUDA preflight failed: {exc}") from exc
    if not available:
        raise IntegrationUnavailable("CUDA is unavailable; refusing an unintended CPU index build.")


def prepare_codelens_index(*, repo_key: str, repograph_path: str, repograph_root: str,
                           codelens_home: str, vectorstore_root: str) -> Path:
    """Build the real CodeLens FAISS index before accepting a long-running task."""
    graph_dir = _inside(Path(repograph_root), repograph_path)
    if not all((graph_dir / name).is_file() for name in ("graph.pkl", "tags.json")):
        raise IntegrationUnavailable("Repograph must contain graph.pkl and tags.json.")
    source_root = Path(codelens_home).resolve()
    if not (source_root / "run_issue_localization.py").is_file():
        raise IntegrationUnavailable("CodeLens source is not mounted at CODELENS_HOME.")
    _check_embedding_device()
    sys.path.insert(0, str(source_root))
    try:
        source_config = importlib.import_module("config")
        store_module = importlib.import_module("agent.memory.unified_vectorstore")
        model = source_config.CODE_EMBEDDING_MODEL
        index_path = vectorstore_path_for(vectorstore_root, repo_key, graph_dir, model)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store = store_module.get_shared_unified_vector_store(str(index_path), model)
        store.ensure_init(repograph_dir=str(graph_dir), repo_key=repo_key)
        return index_path
    finally:
        sys.path.remove(str(source_root))


def run_localization(
    *,
    issue_text: str,
    repo_key: str,
    repository_path: str,
    repograph_path: str,
    repository_root: str,
    repograph_root: str,
    codelens_home: str,
    vectorstore_root: str,
) -> LocalizationOutput:
    source_root = Path(codelens_home).resolve()
    if not (source_root / "run_issue_localization.py").is_file():
        raise IntegrationUnavailable(
            "CodeLens source is not mounted at CODELENS_HOME; expected run_issue_localization.py."
        )
    repo_dir = _inside(Path(repository_root), repository_path)
    graph_dir = _inside(Path(repograph_root), repograph_path)
    missing = [name for name in ("graph.pkl", "tags.json") if not (graph_dir / name).is_file()]
    if missing:
        raise IntegrationUnavailable(
            "Repograph is incomplete; missing " + ", ".join(missing) + "."
        )
    _check_embedding_device()

    if not os.getenv("LLM_PROVIDER"):
        raise IntegrationUnavailable("Set LLM_PROVIDER explicitly; research source defaults conflict.")
    if os.getenv("LLM_PROVIDER") == "ollama" and not os.getenv("OLLAMA_URL"):
        raise IntegrationUnavailable("Set OLLAMA_URL to the actual Qwen service reachable from the worker.")
    required = ("langchain_core", "langchain_openai", "langchain_community", "langchain_huggingface", "sentence_transformers", "faiss", "networkx")
    absent = [name for name in required if importlib.util.find_spec(name) is None]
    if absent:
        raise IntegrationUnavailable("Research Python dependencies are missing: " + ", ".join(absent))

    # Import the inspected source directly. No output is synthesized if import,
    # model access, or inference fails.
    sys.path.insert(0, str(source_root))
    try:
        pipeline_module = importlib.import_module("agent.agents.pipeline")
        metrics = importlib.import_module("agent.utils.localization_metrics")
        source_config = importlib.import_module("config")

        embedding_model = source_config.CODE_EMBEDDING_MODEL
        pipeline_config = pipeline_module.PipelineConfig(
            repograph_dir=str(graph_dir),
            repo_key=repo_key,
            vectorstore_path=str(vectorstore_path_for(vectorstore_root, repo_key, graph_dir, embedding_model)),
            embeddings_model=embedding_model,
            hops=2,
            seed_k=120,
            max_context_chars=8000,
            k_top=20,
            max_reflect_rounds=2,
            policy_driven=True,
            policy_max_steps=8,
            policy_max_llm_calls=8,
            policy_max_wall_time_seconds=1200,
            policy_stop_probability=0.82,
            policy_stable_steps=2,
            policy_target_granularity="line",
            policy_initial_outside_probability=0.20,
            policy_expansion_outside_recovery=0.50,
            policy_open_world_recovery_gain=0.35,
            source_repository_root=str(repo_dir),
        )
        Path(vectorstore_root).mkdir(parents=True, exist_ok=True)
        tag_rows = metrics.load_repograph_tag_objects(graph_dir / "tags.json")
        pipeline = pipeline_module.LocalizationPipeline(pipeline_config)
        result = pipeline.run_sync(
            instance_id=repo_key,
            issue_text=issue_text,
            tag_rows=tag_rows,
            few_shots_long_term="",
        )
        policy = result.policy_trajectory
        retrieval = [
            {
                "granularity": str(step.get("granularity", "")),
                "retrieval_meta": step.get("retrieval_meta", {}),
            }
            for step in (result.graph_context_history or [])
            if isinstance(step, dict)
        ]
        return LocalizationOutput(
            candidates=list(result.best_candidates or []),
            policy_trajectory=policy,
            retrieval_summary=retrieval,
            raw_output=result.best_localization_output or "",
        )
    except IntegrationUnavailable:
        raise
    except Exception as exc:
        raise LocalizationRunError(
            f"CodeLens localization failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass

