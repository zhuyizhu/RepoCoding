from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Set, List, Tuple

from agent.agents.models import GraphContext, GraphEdge, GraphNode


@dataclass
class GraphData:
    """
    A minimal undirected graph abstraction for graphRAG.

    Note: This is intentionally lightweight so we can fall back to rebuilding
    from `tags.json` even when `networkx` is unavailable for unpickling.
    """

    nodes: Dict[str, Dict[str, Any]]
    # Undirected adjacency for robust "ego subgraph" traversal.
    neighbors: Dict[str, Set[str]]
    source: str = "unknown"
    # Optional typed edge list. Existing loaders can leave this empty while the
    # upgraded CriticAgent/RetrievalAgent path can preserve edge type and direction.
    edges: List[GraphEdge] = field(default_factory=list)

    def typed_nodes(self, repo_key: str = "") -> List[GraphNode]:
        """Return legacy node dictionaries as GraphNode models."""
        out: List[GraphNode] = []
        for node_id, payload in self.nodes.items():
            data = dict(payload)
            data.setdefault("node_id", str(node_id))
            if repo_key:
                data.setdefault("repo_key", repo_key)
            out.append(GraphNode.model_validate(data))
        return out

    def to_graph_context(
        self,
        *,
        repo_key: str = "",
        granularity: str = "file",
        text_view: str = "",
    ) -> GraphContext:
        """Build the structured context used by the upgraded agent data flow."""
        return GraphContext(
            repo_key=repo_key,
            granularity=granularity,
            nodes=self.typed_nodes(repo_key=repo_key),
            edges=list(self.edges),
            text_view=text_view,
            retrieval_meta={"source": self.source, "node_count": len(self.nodes), "edge_count": len(self.edges)},
        )

    def ego_nodes_with_dist(self, center: str, radius: int) -> List[Tuple[str, int]]:
        if radius < 0:
            return []
        if center not in self.nodes:
            return []

        dist: Dict[str, int] = {center: 0}
        queue: List[str] = [center]
        while queue:
            n = queue.pop(0)
            d = dist[n]
            if d >= radius:
                continue
            for m in self.neighbors.get(n, set()):
                if m not in dist:
                    dist[m] = d + 1
                    queue.append(m)

        # Sorted for stable prompt context.
        items = sorted(dist.items(), key=lambda x: (x[1], x[0]))
        return items

