from __future__ import absolute_import, annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple, Any

from dataclasses import dataclass, field

from ..models.graph import Edge, Graph, Node
from ..services.similarity import Similarity

logger = logging.getLogger("recap")


@dataclass
class NodeMapping(object):
    """Store query and case node"""

    nq: Node
    nc: Node

    def __eq__(self, other):
        return self.nc.id_ == other.nc.id_ and self.nq.id_ == other.nq.id_


@dataclass
class EdgeMapping(object):
    """Store query and case edge"""

    eq: Edge
    ec: Edge

    def __eq__(self, other):
        return self.ec.id_ == other.ec.id_ and self.eq.id_ == other.eq.id_


@dataclass
class Mapping(object):
    """Store all mappings and perform integrity checks on them"""

    node_mappings: List[NodeMapping] = field(default_factory=list)
    edge_mappings: List[EdgeMapping] = field(default_factory=list)
    similarity: float = field(default=0.0, init=False)

    def _is_node_mapped(self, nc: Node) -> bool:
        """Check if given node is already mapped"""

        nc_list = [node.nc.id_ for node in self.node_mappings]

        return nc.id_ in nc_list

    def _is_edge_mapped(self, ec: Edge) -> bool:
        """Check if given edge is already mapped"""

        ec_list = [edge.ec.id_ for edge in self.edge_mappings]

        return ec.id_ in ec_list

    def _are_nodes_mapped(self, nq: Node, nc: Node) -> bool:
        """Check if the two given nodes are mapped to each other"""

        mapping = NodeMapping(nq, nc)

        return mapping in self.node_mappings

    def is_legal_mapping(self, q: Any, c: Any) -> bool:
        """Check if mapping is legal"""

        if isinstance(q, Node):
            return self.is_legal_node_mapping(q, c)
        elif isinstance(q, Edge):
            return self.is_legal_edge_mapping(q, c)
        return False

    def is_legal_node_mapping(self, nq: Node, nc: Node) -> bool:
        """Check if mapping is legal"""

        is_legal_mapping = True

        if self._is_node_mapped(nc) or nc.type_ != nq.type_:
            is_legal_mapping = False

        return is_legal_mapping

    def is_legal_edge_mapping(self, eq: Edge, ec: Edge) -> bool:
        """Check if mapping is legal"""

        is_legal_mapping = True

        if (
            self._is_edge_mapped(ec)
            or (
                ec.from_node.type_ != eq.from_node.type_
                and ec.to_node.type_ != eq.to_node.type_
            )
            or (
                not self._are_nodes_mapped(eq.from_node, ec.from_node)
                and not self._are_nodes_mapped(eq.to_node, ec.to_node)
            )
        ):
            is_legal_mapping = False

        return is_legal_mapping

    def map(self, q: Any, c: Any) -> None:
        """Create a new mapping"""

        if isinstance(q, Node):
            self.map_nodes(q, c)
        elif isinstance(q, Edge):
            self.map_edges(q, c)

    def map_nodes(self, nq: Node, nc: Node) -> None:
        """Create new node mapping"""

        self.node_mappings.append(NodeMapping(nq, nc))

    def map_edges(self, eq: Edge, ec: Edge) -> None:
        """Create new edge mapping"""

        self.edge_mappings.append(EdgeMapping(eq, ec))

    def get_similarity(self, n_nodes: int, n_edges: int) -> float:
        """Compute similarity for all edge and node mappings"""

        if not self.similarity:
            node_sim = 0.0
            edge_sim = 0.0
            similarity = Similarity.get_instance()

            for nm in self.node_mappings:
                node_sim += similarity.node_similarity(nm.nc, nm.nq)

            for em in self.edge_mappings:
                edge_sim += similarity.edge_similarity(em.ec, em.eq)

            self.similarity = (node_sim + edge_sim) / (n_nodes + n_edges)

        return self.similarity


class SearchNode(object):
    """Specific search node"""

    def __init__(
        self,
        nodes: Dict[int, Node],
        edges: Dict[int, Edge],
        mapping_old: Mapping = None,
    ) -> None:
        self.nodes = copy.copy(nodes)
        self.edges = copy.copy(edges)
        self.f = 1.0

        if mapping_old:
            self.mapping = Mapping(
                copy.copy(mapping_old.node_mappings),
                copy.copy(mapping_old.edge_mappings),
            )
        else:
            self.mapping = Mapping()

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

    def remove(self, key_q: int, x_q: Any) -> None:
        if isinstance(x_q, Node):
            del self.nodes[key_q]
        elif isinstance(x_q, Edge):
            del self.edges[key_q]
