from __future__ import annotations

import json
import logging
import os
import re
import gc
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from agent.memory.utils import safe_truncate
from agent.repograph.java_path_filters import is_test_or_benchmark_java_path
from agent.repograph.repograph_loader import load_repograph, normalize_node_text
from agent.retrieval.path_rerank import path_priority_for_localization
from agent.utils.memory_guard import ensure_vectorstore_load_is_safe

logger = logging.getLogger(__name__)

_SHARED_STORES: "OrderedDict[str, UnifiedVectorStore]" = OrderedDict()
_SHARED_EMBEDDINGS: Dict[str, HuggingFaceEmbeddings] = {}


def _resolve_embedding_device(raw: str) -> str:
    """
    将 ``CODE_EMBEDDING_DEVICE`` 解析为 SentenceTransformer 的 ``device`` 参数。

    ``cuda`` / ``cuda:0`` 在 PyTorch 未编译 CUDA 或不可用时回退到 ``cpu``，避免静默卡死。
    """
    d = (raw or "cpu").strip().lower()
    if not d or d == "cpu":
        return "cpu"
    if d.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "CODE_EMBEDDING_DEVICE=%s 但当前环境 CUDA 不可用，已回退到 cpu",
                    raw.strip() or d,
                )
                return "cpu"
        except Exception as exc:
            logger.warning("检测 CUDA 失败 (%s)，使用 cpu", exc)
            return "cpu"
    return raw.strip()


def _hf_embedding_settings_from_env() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    从环境变量构造 ``HuggingFaceEmbeddings`` 的 ``model_kwargs`` / ``encode_kwargs``。

    - ``CODE_EMBEDDING_DEVICE``: 如 ``cuda``、``cuda:1``、``mps``（默认 ``cpu``）。
    - ``CODE_EMBEDDING_BATCH_SIZE``: ``encode`` 批大小；GPU 上通常可 32–256，视显存而定。
    """
    model_kwargs: Dict[str, Any] = {}
    if os.getenv("CODE_EMBEDDING_TRUST_REMOTE_CODE", "").lower() in ("1", "true", "yes"):
        model_kwargs["trust_remote_code"] = True

    device_env = (os.getenv("CODE_EMBEDDING_DEVICE") or "").strip()
    if device_env:
        model_kwargs["device"] = _resolve_embedding_device(device_env)

    encode_kwargs: Dict[str, Any] = {}
    bs_raw = (os.getenv("CODE_EMBEDDING_BATCH_SIZE") or "").strip()
    if bs_raw.isdigit():
        encode_kwargs["batch_size"] = max(1, int(bs_raw))
    elif str(model_kwargs.get("device", "cpu")).lower().startswith("cuda"):
        encode_kwargs.setdefault("batch_size", 64)

    return model_kwargs, encode_kwargs


def get_shared_unified_vector_store(
    vectorstore_path: str,
    embeddings_model: str,
    *,
    rebuild: bool = False,
) -> "UnifiedVectorStore":
    """
    进程内单例：同一 (vectorstore_path, model, rebuild, HF 设备/编码参数) 只加载一份嵌入与 FAISS，
    避免 memory_few_shots 与 GraphRAG 各建一套模型导致双倍显存/内存与二次磁盘加载。
    """
    mk, ek = _hf_embedding_settings_from_env()
    sig = json.dumps({"mk": mk, "ek": ek}, sort_keys=True, ensure_ascii=False)
    cache_key = f"{vectorstore_path}\0{embeddings_model}\0{int(rebuild)}\0{sig}"
    existing = _SHARED_STORES.get(cache_key)
    if existing is not None:
        _SHARED_STORES.move_to_end(cache_key)
        return existing

    # Run this before constructing/loading the embedding model.  On an 8 GiB
    # WSL instance the former 4.7 GiB cross-repository index used enough RAM to
    # kill the entire VM once VS Code Server was also connected.
    ensure_vectorstore_load_is_safe(vectorstore_path)

    embedding_key = f"{embeddings_model}\0{sig}"
    embeddings = _SHARED_EMBEDDINGS.get(embedding_key)
    if embeddings is None:
        show_prog = os.getenv("CODE_EMBEDDING_SHOW_PROGRESS", "").lower() in ("1", "true", "yes")
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs=mk or {},
            encode_kwargs=ek or {},
            show_progress=show_prog,
        )
        _SHARED_EMBEDDINGS[embedding_key] = embeddings

    store = UnifiedVectorStore(
        UnifiedStoreConfig(
            vectorstore_path=vectorstore_path,
            embeddings_model=embeddings_model,
            rebuild=rebuild,
            model_kwargs=mk or None,
            encode_kwargs=ek or None,
        ),
        embeddings=embeddings,
    )

    try:
        cache_limit = max(1, int(os.getenv("JLOC_VECTORSTORE_CACHE_SIZE", "1") or "1"))
    except ValueError:
        cache_limit = 1
    while len(_SHARED_STORES) >= cache_limit:
        _, evicted = _SHARED_STORES.popitem(last=False)
        evicted.release()
    _SHARED_STORES[cache_key] = store
    return store


@dataclass
class UnifiedStoreConfig:
    vectorstore_path: str
    embeddings_model: str = "BAAI/bge-m3"
    rebuild: bool = False
    # e.g. {"trust_remote_code": True, "device": "cuda"} for SentenceTransformer
    model_kwargs: Optional[Dict[str, Any]] = None
    # e.g. {"batch_size": 128} passed to SentenceTransformer.encode
    encode_kwargs: Optional[Dict[str, Any]] = None


class UnifiedVectorStore:
    """
    A single FAISS vector store that contains:
      - code_node documents built from repograph (graphRAG)
      - issue_example documents added after each localization run (long-term memory few-shot)
    """

    def __init__(self, config: UnifiedStoreConfig, *, embeddings: Optional[HuggingFaceEmbeddings] = None):
        self.config = config
        if embeddings is None:
            mkw = dict(config.model_kwargs) if config.model_kwargs else {}
            ekw = dict(config.encode_kwargs) if config.encode_kwargs else {}
            show_prog = os.getenv("CODE_EMBEDDING_SHOW_PROGRESS", "").lower() in ("1", "true", "yes")
            embeddings = HuggingFaceEmbeddings(
                model_name=config.embeddings_model,
                model_kwargs=mkw,
                encode_kwargs=ekw,
                show_progress=show_prog,
            )
        self._embeddings = embeddings
        self._vs: Optional[FAISS] = None

    def release(self) -> None:
        """Release native FAISS memory when an LRU shard is no longer active."""
        self._vs = None
        gc.collect()

    @property
    def vs(self) -> FAISS:
        if self._vs is None:
            raise RuntimeError("Vectorstore not initialized. Call ensure_init() first.")
        return self._vs

    def _is_initialized(self) -> bool:
        p = Path(self.config.vectorstore_path)
        # FAISS local save creates multiple files; existence of directory is enough.
        if not p.exists():
            return False
        if p.is_dir():
            try:
                return any(p.iterdir())
            except Exception:
                return True
        # If user configured FAISS with a non-directory path, treat as initialized.
        return True

    def ensure_init(self, repograph_dir: str, repo_key: str) -> FAISS:
        """
        初始化或追加当前 ``repo_key`` 的 code_node 向量。

        注意：不得在 ``self._vs`` 已存在时无条件 ``return``，否则批处理第二条实例时
        永远不会把新 ``repo_key``/新 ``repograph_dir`` 的节点写入 FAISS。
        """
        vectorstore_dir = Path(self.config.vectorstore_path)
        manifest_path = vectorstore_dir / "manifest.json"
        manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        already_added = (manifest.get("repo_keys", []) or []) if isinstance(manifest.get("repo_keys", []), list) else []
        repo_already_added = repo_key in already_added

        if self._vs is None and (not self.config.rebuild) and self._is_initialized():
            ensure_vectorstore_load_is_safe(self.config.vectorstore_path)
            logger.info(
                "Loading FAISS from disk path=%s (may take a minute for large indexes)",
                self.config.vectorstore_path,
            )
            print(
                f"[vectorstore] Loading FAISS index from {self.config.vectorstore_path} …",
                flush=True,
            )
            self._vs = FAISS.load_local(
                self.config.vectorstore_path,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            print("[vectorstore] FAISS load_local done.", flush=True)

        if self._vs is not None and repo_already_added:
            return self._vs

        logger.info(
            "Building/appending code_node embeddings repo_key=%s repograph=%s",
            repo_key,
            repograph_dir,
        )
        print(
            f"[vectorstore] Indexing code nodes for repo_key={repo_key} (manifest had it: {repo_already_added}) …",
            flush=True,
        )

        repograph = load_repograph(repograph_dir)
        try:
            build_chunk_size = max(
                32,
                int(os.getenv("JLOC_VECTORSTORE_BUILD_CHUNK_SIZE", "512") or "512"),
            )
        except ValueError:
            build_chunk_size = 512

        docs: List[Document] = []
        n_docs = 0

        def flush_docs() -> None:
            nonlocal docs, n_docs
            if not docs:
                return
            if self._vs is None:
                self._vs = FAISS.from_documents(docs, self._embeddings)
            else:
                self._vs.add_documents(docs)
            n_docs += len(docs)
            docs = []

        for node_id, node_data in repograph.nodes.items():
            rel_fname = str(node_data.get("rel_fname") or node_data.get("fname") or "")
            if is_test_or_benchmark_java_path(rel_fname.replace("\\", "/")):
                continue
            page_content = normalize_node_text(node_id, node_data)
            docs.append(
                Document(
                    page_content=page_content[:8000],
                    metadata={
                        "doc_type": "code_node",
                        "repo_key": repo_key,
                        "node_id": node_id,
                        "rel_fname": rel_fname,
                    },
                )
            )
            if len(docs) >= build_chunk_size:
                flush_docs()
        flush_docs()

        if n_docs > 3000:
            dev = str((self.config.model_kwargs or {}).get("device", "cpu"))
            hint = (
                "GPU 上通常明显更快：可先 export CODE_EMBEDDING_DEVICE=cuda "
                "（可选 CODE_EMBEDDING_BATCH_SIZE=128）。"
                if not str(dev).lower().startswith("cuda")
                else "当前使用 GPU 嵌入。"
            )
            print(
                f"[vectorstore] {n_docs} code_node documents to embed（设备≈{dev}；{hint}）…",
                flush=True,
            )

        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        if self._vs is None:
            raise RuntimeError(f"No indexable code nodes found in Repograph: {repograph_dir}")
        self._vs.save_local(self.config.vectorstore_path)

        try:
            if isinstance(manifest.get("repo_keys", []), list):
                repo_keys = list(manifest.get("repo_keys", []))
            else:
                repo_keys = []
            if repo_key not in repo_keys:
                repo_keys.append(repo_key)
            manifest["repo_keys"] = repo_keys
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        print(f"[vectorstore] Indexed {n_docs} code nodes for {repo_key}.", flush=True)
        return self._vs

    def add_issue_example(
        self,
        repo_key: str,
        issue_text: str,
        localization_text: str,
    ) -> None:
        doc = Document(
            page_content=(
                "ISSUE_EXAMPLE\n"
                f"RepoKey: {repo_key}\n"
                f"Issue:\n{safe_truncate(issue_text, 4000)}\n\n"
                f"Localization:\n{safe_truncate(localization_text, 4000)}"
            ),
            metadata={
                "doc_type": "issue_example",
                "repo_key": repo_key,
            },
        )
        self.vs.add_documents([doc])
        self.vs.save_local(self.config.vectorstore_path)

    def search_code_nodes(self, issue_text: str, repo_key: str, k: int) -> List[Document]:
        """
        Hybrid retrieval: over-fetch then rerank by (distance - path_prior) so core library
        files beat androidTest/benchmark noise when distances are similar.
        """
        fetch_k = max(k * 8, k + 80, 120)
        try:
            pairs = self.vs.similarity_search_with_score(issue_text, k=fetch_k)
        except Exception:
            pairs = [(d, 0.0) for d in self.vs.similarity_search(issue_text, k=fetch_k)]

        scored: List[tuple[float, Document]] = []
        for d, dist in pairs:
            if d.metadata.get("doc_type") != "code_node":
                continue
            if d.metadata.get("repo_key") != repo_key:
                continue
            rel = str(d.metadata.get("rel_fname") or "")
            if rel and is_test_or_benchmark_java_path(rel.replace("\\", "/")):
                continue
            if not rel and d.page_content:
                m = re.search(r"(?m)^File:\s*(.+)$", d.page_content)
                if m:
                    rel = m.group(1).strip()
            prior = path_priority_for_localization(rel)
            effective = float(dist) - prior
            scored.append((effective, d))

        scored.sort(key=lambda x: x[0])
        out: List[Document] = []
        seen_nid: Set[str] = set()
        for _, d in scored:
            nid = d.metadata.get("node_id")
            key = str(nid) if nid else str(id(d))
            if key in seen_nid:
                continue
            seen_nid.add(key)
            out.append(d)
            if len(out) >= k:
                break
        return out

    def search_issue_examples(self, issue_text: str, repo_key: str, k: int) -> List[Document]:
        docs = self.vs.similarity_search(issue_text, k=k)
        filtered: List[Document] = []
        for d in docs:
            if d.metadata.get("doc_type") != "issue_example":
                continue
            if d.metadata.get("repo_key") != repo_key:
                continue
            filtered.append(d)
        return filtered
