"""G: repository observations, independent of any agent or scheduling policy."""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

from agent.agents.models import GraphContext, GraphNode
from agent.agents.retrieval_agent import RetrievalAgent


class CodeContextTool(RetrievalAgent):
    """Compatibility adapter around existing GraphRAG plus bounded source reads."""

    def __init__(self, *args, source_repository_root="", allow_unconstrained_search=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_unconstrained_search = bool(allow_unconstrained_search)
        parts = self.repo_key.rsplit("__", 1)
        self.revision = parts[-1] if len(parts) == 2 else ""
        self.repository = None
        if source_repository_root:
            root = Path(source_repository_root)
            repo_name = parts[0]
            candidates = [root / (repo_name + ".git"), root / repo_name]
            # Older Java runs retained ordinary checkouts under the repository
            # name (for example, jackson-databind) rather than bare mirrors.
            if "__" in repo_name:
                candidates.append(root / repo_name.rsplit("__", 1)[-1])
            self.repository = next(
                (candidate for candidate in candidates
                 if candidate.is_dir() and
                 (candidate.name.endswith(".git") or (candidate / ".git").exists())),
                None,
            )
        self.extra_contexts = []
        self.tool_events = []

    def _source(self, path, start=1, count=60):
        if not self.repository or not self.repository.is_dir():
            return {"status": "unavailable", "reason": "source_repository_missing"}
        if not self.revision or not path.endswith(".java") or path.startswith("/") or ".." in Path(path).parts:
            return {"status": "unavailable", "reason": "invalid_source_request"}
        command = ["git", "-C", str(self.repository), "show", f"{self.revision}:{path}"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if result.returncode:
            return {"status": "unavailable", "reason": "path_not_present_at_base_revision", "file": path}
        lines = result.stdout.splitlines()
        start = max(1, min(int(start), max(1, len(lines))))
        end = min(len(lines), start + max(1, min(int(count), 80)) - 1)
        text = "\n".join(f"{i}: {lines[i-1]}" for i in range(start, end + 1))
        return {"status": "ok", "file": path, "revision": self.revision,
                "start": start, "end": end, "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest()}

    async def execute(self, request, context, issue_text):
        """Execute a model-selected allowlisted operation; never execute model code."""
        tool = str(request.get("tool", ""))
        updated = context.model_copy(deep=True)
        if tool == "read_source":
            observation = await asyncio.to_thread(
                self._source, str(request.get("file", "")),
                request.get("start", 1), request.get("count", 60))
            if observation["status"] == "ok":
                path = observation["file"]
                excerpt = f"File: {path}\nBase revision: {self.revision}\nNumbered source:\n{observation['text']}"
                source_nodes = [
                    GraphNode(node_id=f"source::{path}:{line}", file_path=path,
                              category="statement", kind="unknown", granularity="line",
                              line_start=line, line_end=line)
                    for line in range(observation["start"], observation["end"] + 1)
                ]
                extra = GraphContext(
                    context_id=observation["sha256"], granularity=context.granularity,
                    nodes=source_nodes, text_view=excerpt,
                    valid_locations={path: list(range(observation["start"], observation["end"] + 1))})
                self.extra_contexts.append(extra)
                updated.nodes.extend(source_nodes)
                updated.valid_locations[path] = sorted(set(updated.valid_locations.get(path, [])) |
                                                        set(extra.valid_locations[path]))
                cap = self.max_context_chars or 12000
                updated.text_view = (excerpt + "\n\n" + context.text_view)[:cap]
        elif tool == "graph_neighbors":
            node_id = str(request.get("node_id", ""))
            edges = [e for e in context.edges if e.src == node_id or e.dst == node_id]
            observation = {"status": "ok" if edges else "unobserved", "node_id": node_id,
                           "edges": [e.model_dump() for e in edges[:30]],
                           "note": "Only recorded edges; absence is not evidence of non-reachability."}
        elif tool == "search":
            query = str(request.get("query", "")).strip()[:600]
            if not self.allow_unconstrained_search:
                observation = {"status": "unavailable", "reason": "scope_recovery_disabled"}
            elif not query:
                observation = {"status": "unavailable", "reason": "empty_query"}
            else:
                key, _, meta = await self.retrieve(
                    issue_text=issue_text + "\nSearch focus: " + query,
                    granularity=context.granularity, force_file_paths=None,
                    hops_override=self.hops + 1)
                found = self.get_structured_context(key)
                observation = {"status": "ok", "query": query, "context_id": key,
                               "node_count": len(found.nodes) if found else 0}
                if found is not None:
                    self.extra_contexts.append(found)
                    updated = self._merge(updated, found)
                    cap = self.max_context_chars or 12000
                    updated.text_view = (found.text_view + "\n\n" + context.text_view)[:cap]
        else:
            observation = {"status": "unavailable", "reason": "unknown_tool", "tool": tool}
        event = {"request": request, "observation": observation}
        self.tool_events.append(event)
        return updated, event

    @staticmethod
    def _merge(base, extra):
        base = base.model_copy(deep=True)
        base.nodes = list({n.node_id: n for n in [*base.nodes, *extra.nodes]}.values())
        base.edges = list({(e.src, e.dst, e.edge_type): e for e in [*base.edges, *extra.edges]}.values())
        for path, lines in extra.valid_locations.items():
            base.valid_locations[path] = sorted(set(base.valid_locations.get(path, [])) | set(lines))
        return base

    def merge_structured_contexts(self, *args, **kwargs):
        context = super().merge_structured_contexts(*args, **kwargs)
        for extra in self.extra_contexts:
            context = self._merge(context, extra)
        return context
