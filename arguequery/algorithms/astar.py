from __future__ import absolute_import, annotations

import bisect
import logging
import multiprocessing
import random
import typing as t
from abc import ABC
from dataclasses import dataclass, field
from typing import List

import arguebuf as ag
from arguequery.config import config
from arguequery.models.mapping import FacMapping, FacResults
from arguequery.services import nlp

logger = logging.getLogger("recap")


@dataclass(frozen=True, eq=True)
class GenericMapping(ABC):
    query: t.Union[ag.Node, ag.Edge]
    case: t.Union[ag.Node, ag.Edge]

    def similarity(self) -> float:
        return nlp.similarity(self.query, self.case)


@dataclass(frozen=True, eq=True)
class NodeMapping(GenericMapping):
    """Store query and case node"""

    query: ag.Node
    case: ag.Node


@dataclass(frozen=True, eq=True)
class EdgeMapping(GenericMapping):
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

    def similarity(self) -> float:
        """Compute similarity for all edge and node mappings

        We are deliberately not using a standard mean here!
        As the mapping may be uncomplete, we need to divide by the total number of nodes/edges.
        """

        return sum(
            mapping.similarity()
            for mapping in self.node_mappings.union(self.edge_mappings)
        ) / (self.available_nodes + self.available_edges)


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


def _init_mp():
    # Each worker needs its own client.
    # Otherwise, the client will be protected by a mutex, resulting in worse runtimes.
    nlp.client = nlp.init_client()


def run(cases: t.Mapping[str, ag.Graph], query: ag.Graph) -> FacResults:
    similarities: t.Dict[str, float] = {}
    mappings: t.Dict[str, t.Set[FacMapping]] = {}

    params = [
        (
            case_graph,
            case_id,
            query,
            i,
            len(cases),
            config.cbr.queue_limit,
        )
        for (i, (case_id, case_graph)) in enumerate(cases.items())
    ]

    logger.info(f"A* Search for query '{query.name}'.")

    if config.debug:
        results = [a_star_search(*param) for param in params]
    else:
        # If we do no 'reset' the cache, it is copied to every multiprocess.
        # As this is quite slow for large vectors, we just reset it.
        nlp.vector_cache = {}

        with multiprocessing.Pool(initializer=_init_mp) as pool:
            results = pool.starmap(a_star_search, params)

    for case_id, mapping in results:
        similarities[case_id] = mapping.similarity()
        mappings[case_id] = {
            FacMapping(entry.query.id, entry.case.id, entry.similarity())
            for entry in mapping.node_mappings
        }

    return FacResults(similarities, mappings)


# According to Bergmann and Gil, 2014
def a_star_search(
    case: ag.Graph,
    case_id: str,
    query: ag.Graph,
    current_iteration: int,
    total_iterations: int,
    queue_limit: int,
) -> t.Tuple[str, Mapping]:
    """Perform an A* analysis of the case base and the query"""
    q: List[SearchNode] = []
    s0 = SearchNode(
        len(query.nodes),
        len(query.edges),
        query.nodes.values(),
        query.edges.values(),
    )

    bisect.insort(q, s0)

    while q[-1].nodes or q[-1].edges:
        q = _expand(q, case, query, queue_limit)

    candidate = q[-1]

    logger.debug(
        f"A* search for {case.name} finished. ({current_iteration}/{total_iterations})"
    )

    return (case_id, candidate.mapping)


def _expand(
    q: List[SearchNode], case: ag.Graph, query: ag.Graph, queue_limit: int
) -> List[SearchNode]:
    """Expand a given node and its queue"""

    s = q[-1]
    mapped = False
    query_obj, iterator = select1(s, query, case)

    if query_obj and iterator:
        for case_obj in iterator:
            if s.mapping.is_legal_mapping(query_obj, case_obj):
                s_new = SearchNode(
                    len(query.nodes),
                    len(query.edges),
                    s.nodes,
                    s.edges,
                    s.mapping,
                )
                s_new.mapping.map(query_obj, case_obj)
                s_new.remove(query_obj)
                s_new.f = g(s_new, query) + h2(s_new, query, case)
                bisect.insort(q, s_new)
                mapped = True

        if mapped:
            q.remove(s)
        else:
            s.remove(query_obj)

    return q[len(q) - queue_limit :] if queue_limit > 0 else q


def select1(
    s: SearchNode, query: ag.Graph, case: ag.Graph
) -> t.Tuple[
    t.Optional[t.Union[ag.Node, ag.Edge, None]],
    t.Optional[t.Iterable[t.Union[ag.Node, ag.Edge]]],
]:
    query_obj = None
    candidates = None

    if s.nodes:
        query_obj = random.choice(tuple(s.nodes))
        candidates = (
            case.atom_nodes.values()
            if isinstance(query_obj, ag.AtomNode)
            else case.scheme_nodes.values()
        )
    elif s.edges:
        query_obj = random.choice(tuple(s.edges))
        candidates = case.edges.values()

    return query_obj, candidates


# def select2(s: SearchNode, query: ag.Graph, case: ag.Graph) -> Tuple:
#     pass


def h1(s: SearchNode, query: ag.Graph, case: ag.Graph) -> float:
    """Heuristic to compute future costs"""

    return (len(s.nodes) + len(s.edges)) / (len(query.nodes) + len(query.edges))


def h2(s: SearchNode, query: ag.Graph, case: ag.Graph) -> float:
    h_val = 0

    for x in s.nodes:
        max_sim = max(
            nlp.similarity(x, y)
            for y in (
                case.atom_nodes.values()
                if isinstance(x, ag.AtomNode)
                else case.scheme_nodes.values()
            )
        )

        h_val += max_sim

    for x in s.edges:
        max_sim = max(nlp.similarity(x, y) for y in case.edges.values())

        h_val += max_sim

    return h_val / (len(query.nodes) + len(query.edges))


def g(s: SearchNode, query: ag.Graph) -> float:
    """Function to compute the costs of all previous steps"""

    return s.mapping.similarity()
