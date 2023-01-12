from __future__ import absolute_import, annotations

import logging
import typing as t

import arguebuf as ag

from arguequery.algorithms import astar, isomorphism
from arguequery.models.mapping import FacResults
from arguequery.services import nlp
from arguequery.types import MappingAlgorithm

logger = logging.getLogger("recap")


def mac(
    cases: t.Mapping[str, ag.Graph], query: t.Union[str, ag.Graph]
) -> t.Dict[str, float]:
    similarities = nlp.similarities((case, query) for case in cases.values())
    return dict(zip(cases.keys(), similarities))


def fac(
    cases: t.Mapping[str, ag.Graph], query: ag.Graph, algorithm: MappingAlgorithm
) -> FacResults:
    """Perform an in-depth analysis of the prefilter results"""

    if algorithm.startswith("astar"):
        return astar.run(cases, query)
    elif algorithm == "isomorphism":
        return isomorphism.run(cases, query)

    raise ValueError(f"Algorithm {algorithm} not implemented.")
