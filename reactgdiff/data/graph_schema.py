"""Lightweight heterogeneous process graph schema.

The project intentionally keeps the serialized graph format simple:
JSON-compatible node and edge dictionaries with stable IDs. This is enough for
dataset construction, metrics, and HTML visualization before the model-specific
graph tensors are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphNode:
    """A typed process-graph node."""

    id: str
    type: str
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "attrs": self.attrs,
        }


@dataclass(slots=True)
class GraphEdge:
    """A typed directed process-graph edge."""

    id: str
    source: str
    target: str
    type: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "attrs": self.attrs,
        }


@dataclass
class ProcessGraph:
    """Serializable heterogeneous process graph for one OpenExp example."""

    graph_id: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        attrs: dict[str, Any] | None = None,
    ) -> GraphNode:
        node = GraphNode(
            id=node_id,
            type=node_type,
            label=label,
            attrs=dict(attrs or {}),
        )
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        attrs: dict[str, Any] | None = None,
    ) -> GraphEdge:
        edge = GraphEdge(
            id=f"e_{len(self.edges):04d}",
            source=source,
            target=target,
            type=edge_type,
            attrs=dict(attrs or {}),
        )
        self.edges.append(edge)
        return edge

    def has_node(self, node_id: str) -> bool:
        return any(node.id == node_id for node in self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }
