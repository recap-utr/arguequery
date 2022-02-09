from __future__ import absolute_import, annotations

import bisect
import logging
import multiprocessing
import random
import typing as t
from typing import Dict, List, Optional, Tuple, Union

import arguebuf as ag
from arg_services.retrieval.v1 import retrieval_pb2
from arguequery.models.mapping import Mapping, SearchNode
from arguequery.models.result import Result
from arguequery.services import nlp

logger = logging.getLogger("recap")
from arguequery.config import config


def _init_mp():
    # Each worker needs its own client.
    # Otherwise, the client will be protected by a mutex, resulting in worse runtimes.
    nlp.client = nlp.init_client()


def run(cases: t.Mapping[str, ag.Graph], query: ag.Graph) -> t.Dict[str, float]:
    results: List[t.Tuple[str, float]] = []
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

    return dict(results)


# According to Bergmann and Gil, 2014
def a_star_search(
    case: ag.Graph,
    case_id: str,
    query: ag.Graph,
    current_iteration: int,
    total_iterations: int,
    queue_limit: int,
) -> t.Tuple[str, float]:
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

    return (case_id, candidate.mapping.similarity)


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

    return s.mapping.similarity
