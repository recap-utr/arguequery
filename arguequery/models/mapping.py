from __future__ import absolute_import, annotations

import copy
import logging
import typing as t
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import arguebuf as ag
from arguequery.services import nlp

logger = logging.getLogger("recap")


@dataclass(frozen=True, eq=True)
class NodeMapping:
    """Store query and case node"""

    query: ag.Node
    case: ag.Node


@dataclass(frozen=True, eq=True)
class EdgeMapping:
    """Store query and case edge"""

    query: ag.Edge
    case: ag.Edge


@dataclass
class Mapping:
    """Store all mappings and perform integrity checks on them"""

    available_nodes: int
    available_edges: int
    node_mappings: t.Set[NodeMapping] = field(default_factory=set)
    edge_mappings: t.Set[EdgeMapping] = field(default_factory=set)

    def _is_node_mapped(self, nc: ag.Node) -> bool:
        """Check if given node is already mapped"""

        return nc in self.node_mappings

    def _is_edge_mapped(self, ec: ag.Edge) -> bool:
        """Check if given edge is already mapped"""

        return ec in self.edge_mappings

    def _are_nodes_mapped(self, nq: ag.Node, nc: ag.Node) -> bool:
        """Check if the two given nodes are mapped to each other"""

        return NodeMapping(nq, nc) in self.node_mappings

    def is_legal_mapping(
        self, q: t.Union[ag.Node, ag.Edge], c: t.Union[ag.Node, ag.Edge]
    ) -> bool:
        """Check if mapping is legal"""

        if isinstance(q, ag.Node) and isinstance(c, ag.Node):
            return self.is_legal_node_mapping(q, c)
        elif isinstance(q, ag.Edge) and isinstance(c, ag.Edge):
            return self.is_legal_edge_mapping(q, c)
        return False

    def is_legal_node_mapping(self, nq: ag.Node, nc: ag.Node) -> bool:
        """Check if mapping is legal"""

        return not (self._is_node_mapped(nc) or type(nc) != type(nq))

    def is_legal_edge_mapping(self, eq: ag.Edge, ec: ag.Edge) -> bool:
        """Check if mapping is legal"""

        return not (
            self._is_edge_mapped(ec)
            or not self.is_legal_node_mapping(eq.source, ec.source)
            or not self.is_legal_node_mapping(eq.target, ec.target)
        )

    def map(self, q: t.Union[ag.Node, ag.Edge], c: t.Union[ag.Node, ag.Edge]) -> None:
        """Create a new mapping"""

        if isinstance(q, ag.Node) and isinstance(c, ag.Node):
            self.map_nodes(q, c)

        elif isinstance(q, ag.Edge) and isinstance(c, ag.Edge):
            self.map_edges(q, c)

    def map_nodes(self, nq: ag.Node, nc: ag.Node) -> None:
        """Create new node mapping"""

        self.node_mappings.add(NodeMapping(nq, nc))

    def map_edges(self, eq: ag.Edge, ec: ag.Edge) -> None:
        """Create new edge mapping"""

        self.edge_mappings.add(EdgeMapping(eq, ec))

    @property
    def similarity(self) -> float:
        """Compute similarity for all edge and node mappings"""

        node_sim = sum(
            nlp.similarity(mapping.case, mapping.query)
            for mapping in self.node_mappings
        )
        edge_sim = sum(
            nlp.similarity(mapping.case, mapping.query)
            for mapping in self.edge_mappings
        )

        return (node_sim + edge_sim) / (self.available_nodes + self.available_edges)


class SearchNode:
    """Specific search node"""

    def __init__(
        self,
        available_nodes: int,
        available_edges: int,
        nodes: t.Iterable[ag.Node],
        edges: t.Iterable[ag.Edge],
        mapping_old: Mapping = None,
    ) -> None:
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.f = 1.0

        if mapping_old:
            self.mapping = Mapping(
                available_nodes,
                available_edges,
                set(mapping_old.node_mappings),
                set(mapping_old.edge_mappings),
            )
        else:
            self.mapping = Mapping(available_nodes, available_edges)

    def __lt__(self, other) -> bool:
        return self.f < other.f

    def __le__(self, other) -> bool:
        return self.f <= other.f

    def __gt__(self, other) -> bool:
        return self.f > other.f

    def __ge__(self, other) -> bool:
        return self.f >= other.f

    def __eq__(self, other) -> bool:
        return self.f == other.f

    def remove(self, q: t.Union[ag.Node, ag.Edge]) -> None:
        if isinstance(q, ag.Node):
            self.nodes.remove(q)

        elif isinstance(q, ag.Edge):
            self.edges.remove(q)
