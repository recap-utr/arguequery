from __future__ import absolute_import, annotations

import bisect
import logging
import multiprocessing
import random
import typing as t
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import arguebuf as ag
from arg_services.retrieval.v1.retrieval_pb2 import MappingAlgorithm
from arguequery.algorithms import astar, isomorphism
from arguequery.models.mapping import Mapping, SearchNode
from arguequery.models.result import Result
from arguequery.services import nlp

logger = logging.getLogger("recap")
from arguequery.config import config

# @dataclass
# class Mapping:
#     query_id: str
#     case_id: str
#     similarity: float

# @dataclass
# class FacResults:
#     similarities: t.Dict[str, float]
#     mappings: t.Dict[str, Mapping]


def mac(
    cases: t.Mapping[str, ag.Graph], query: t.Union[str, ag.Graph]
) -> t.Dict[str, float]:
    similarities = nlp.similarities((case, query) for case in cases.values())
    return dict(zip(cases.keys(), similarities))


def fac(
    cases: t.Mapping[str, ag.Graph], query: ag.Graph, algorithm: MappingAlgorithm.V
) -> t.Dict[str, float]:
    """Perform an in-depth analysis of the prefilter results"""

    if algorithm == MappingAlgorithm.MAPPING_ALGORITHM_ASTAR:
        return astar.run(cases, query)
    elif algorithm == MappingAlgorithm.MAPPING_ALGORITHM_ISOMORPHISM:
        return isomorphism.run(cases, query)

    raise NotImplementedError("The specified algorithm is currently not supported.")
