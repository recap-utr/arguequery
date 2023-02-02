from __future__ import absolute_import, annotations

import logging
import typing as t

import arguebuf as ag
from arg_services.cbr.v1beta.retrieval_pb2 import MappingAlgorithm

from arguequery.algorithms import astar, isomorphism
from arguequery.models.graph import Graph
from arguequery.models.mapping import FacResults
from arguequery.services.nlp import Nlp

logger = logging.getLogger(__name__)


def mac(cases: t.Mapping[str, Graph], query: Graph, nlp: Nlp) -> t.Dict[str, float]:
    similarities = nlp.similarities((case.text, query.text) for case in cases.values())
    return dict(zip(cases.keys(), similarities))


def fac(
    cases: t.Mapping[str, ag.Graph],
    query: ag.Graph,
    algorithm: MappingAlgorithm.ValueType,
    nlp: Nlp,
    astar_queue_limit,
) -> FacResults:
    """Perform an in-depth analysis of the prefilter results"""

    if algorithm == MappingAlgorithm.MAPPING_ALGORITHM_ASTAR:
        return astar.run(cases, query, nlp, astar_queue_limit)
    elif algorithm == MappingAlgorithm.MAPPING_ALGORITHM_ISOMORPHISM:
        return isomorphism.run(cases, query, nlp)

    raise ValueError(f"Algorithm {algorithm} not implemented.")
