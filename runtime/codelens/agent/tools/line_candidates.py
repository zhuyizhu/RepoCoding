"""Source-grounded line proposals, reusable without a scheduler or an LLM.

Only positions explicitly nominated by an agent enter this ranking. Graph method
entry nodes remain retrieval evidence; they do not become line predictions.
"""
from __future__ import annotations

import asyncio
import hashlib
from agent.agents.models import GraphNode
from agent.policy.evidence import _canonical_file_path


class LineCandidateTool:
    def __init__(self, source_reader):
        self.source_reader = source_reader
        self.candidates = {}
        self.events = []
        self.cache = {}

    @staticmethod
    def resolve_path(path, context):
        paths = {n.file_path for n in context.nodes if n.file_path}
        paths.update(context.valid_locations)
        if path in paths:
            return path
        canonical = _canonical_file_path(path)
        matches = [p for p in paths if _canonical_file_path(p) == canonical]
        if len(matches) != 1:
            matches = [p for p in paths if p.endswith("/" + path.lstrip("/"))]
        return matches[0] if len(matches) == 1 else ""

    async def observe(self, result, context, *, role, max_candidates=5):
        updated = context.model_copy(deep=True)
        snippets = []
        for rank, target in enumerate([result.target, *result.alternatives][:max_candidates], 1):
            if target.target_type != "line" or not target.line_start:
                continue
            path = self.resolve_path(target.file_path, context)
            line = int(target.line_start)
            event = {"role": role, "rank": rank, "requested": target.target,
                     "file": path, "line": line, "accepted": False}
            if not path:
                event["reason"] = "missing_or_ambiguous_path"
                self.events.append(event)
                continue
            cache_key = (path, line)
            if cache_key not in self.cache:
                self.cache[cache_key] = await asyncio.to_thread(
                    self.source_reader, path, max(1, line - 2), 5)
            observation = self.cache[cache_key]
            # _source clamps requests: explicitly reject out-of-file positions.
            grounded = (observation.get("status") == "ok"
                        and observation["start"] <= line <= observation["end"])
            already_observed = any(n.file_path == path and n.line_start == line
                                   for n in context.nodes) and line in context.valid_locations.get(path, [])
            if not grounded and not already_observed:
                event["reason"] = observation.get("reason", "line_not_observed")
                self.events.append(event)
                continue
            key = f"{_canonical_file_path(path)}:{line}"
            item = self.candidates.setdefault(key, {"file": path, "line": line, "votes": {}})
            # Repeated verification cannot accumulate confidence by repetition.
            item["votes"][role] = max(item["votes"].get(role, 0.0), 1.0 / rank)
            event.update(accepted=True, evidence_source="base_revision_source" if grounded else "existing_graph",
                         source=observation)
            self.events.append(event)
            if grounded:
                excerpt = f"File: {path}\nNumbered base-revision source:\n{observation['text']}"
                snippets.append(excerpt)
                node = GraphNode(node_id=f"line-proposal::{path}:{line}", file_path=path,
                    category="statement", kind="unknown", granularity="line",
                    line_start=line, line_end=line, source_text_excerpt=observation["text"],
                    metadata={"line_proposal": True, "source_sha256": observation["sha256"]})
                if node.node_id not in {n.node_id for n in updated.nodes}:
                    updated.nodes.append(node)
                updated.valid_locations[path] = sorted(set(updated.valid_locations.get(path, [])) | {line})
        if snippets:
            limit = max(1, len(context.text_view))
            updated.text_view = ("\n\n".join(dict.fromkeys(snippets)) + "\n\n" + context.text_view)[:limit]
        return updated

    def ranked(self, limit=12):
        return sorted(self.candidates.values(),
                      key=lambda x: (-sum(x["votes"].values()), x["file"], x["line"]))[:limit]

    def render(self, limit=12):
        return "\n".join(
            f"文件: {item['file']} 行: {item['line']} 理由: source_grounded_agent_proposal"
            for item in self.ranked(limit))

    def audit(self):
        return {"schema": "jloc.line_proposals.v1", "events": self.events,
                "ranking": self.ranked(), "source_reads": len(self.cache)}

