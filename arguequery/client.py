from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import arguebuf as ag
import grpc
from arg_services.nlp.v1 import nlp_pb2
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.models.result import Result
from arguequery.services import exporter, nlp, retrieval
from arguequery.services.evaluation import Evaluation

log = logging.getLogger("recap")
log.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

from arguequery.config import config


def main() -> None:
    """Calculate similarity of queries and case base"""

    start_time = 0
    duration = 0
    eval_dict = {}
    evaluations: List[t.Optional[Evaluation]] = []

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(
            config.retrieval_url, [("grpc.lb_policy_name", "round_robin")]
        )
    )

    cases = {
        file.name: ag.Graph.from_file(file)
        for file in Path(config.path.cases).glob(config.path.glob_pattern)
    }
    protobuf_cases = {name: graph.to_protobuf() for name, graph in cases.items()}

    queries = {
        file.name: ag.Graph.from_file(file)
        for file in Path(config.path.queries).glob(config.path.glob_pattern)
    }

    start_time = timer()

    for query_key, query in queries.items():
        req = retrieval_pb2.RetrieveRequest(
            cases=protobuf_cases,
            query_graph=query.to_protobuf(),
            enforce_scheme_types=config.nlp.enforce_scheme_types,
            use_scheme_ontology=config.nlp.use_scheme_ontology,
            limit=config.cbr.limit,
            mac_phase=config.cbr.mac,
            fac_phase=config.cbr.fac,
            mapping_algorithm=retrieval_pb2.MappingAlgorithm.MAPPING_ALGORITHM_ASTAR,
            nlp_config=nlp_pb2.NlpConfig(
                language=config.nlp.language,
                spacy_model="en_core_web_lg",
                similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
            ),
        )
        res: retrieval_pb2.RetrieveResponse = client.Retrieve(req)

        evaluation = None
        mac_export = None
        fac_export = None

        if mac_results := res.mac_ranking:
            mac_export = exporter.get_results(cases, mac_results)
            evaluation = Evaluation(cases, mac_results, query)

        if res.fac_ranking:
            fac_results = [result.case for result in res.fac_ranking]
            fac_export = exporter.get_results(cases, fac_results)
            evaluation = Evaluation(cases, fac_results, query)

        evaluations.append(evaluation)

        if config.export.individual_results:
            exporter.export_results(
                query_key,
                mac_export,
                fac_export,
                evaluation,
            )
            log.info("Individual Results were exported.")

    duration = timer() - start_time
    eval_dict = exporter.get_results_aggregated(evaluations)

    if config.export.aggregated_results:
        exporter.export_results_aggregated(eval_dict, duration, config.as_dict())
        log.info("Aggregated Results were exported.")


if __name__ == "__main__":
    main()
