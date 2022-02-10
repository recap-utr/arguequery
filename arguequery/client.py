from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from typing import List

import arguebuf as ag
import grpc
from arg_services.nlp.v1 import nlp_pb2
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.services import exporter, nlp, retrieval
from arguequery.services.evaluation import Evaluation

log = logging.getLogger("recap")
log.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

from arguequery.config import config

_nlp_configs = {
    "default": nlp_pb2.NlpConfig(
        language=config.nlp.language,
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    )
}


def main() -> None:
    """Calculate similarity of queries and case base"""

    start_time = 0
    duration = 0
    eval_dict = {}
    evaluations: List[t.Optional[Evaluation]] = []

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(
            config.microservices.retrieval, [("grpc.lb_policy_name", "round_robin")]
        )
    )

    cases: t.Dict[Path, ag.Graph] = {
        file: ag.Graph.from_file(file)
        for file in Path(config.path.cases).glob(config.path.case_graphs_pattern)
    }
    protobuf_cases = {
        str(name.relative_to(config.path.cases)): graph.to_protobuf()
        for name, graph in cases.items()
    }

    queries: t.Dict[Path, t.Union[str, ag.Graph]] = {
        file: ag.Graph.from_file(file)
        for file in Path(config.path.queries).glob(config.path.query_graphs_pattern)
    }

    for file in Path(config.path.queries).glob(config.path.glob_pattern):
        with file.open("r", encoding="utf-8") as f:
            queries[file] = f.read()

    start_time = timer()

    for query_file, query in queries.items():
        req = retrieval_pb2.RetrieveRequest(
            cases=protobuf_cases,
            enforce_scheme_types=config.nlp.enforce_scheme_types,
            use_scheme_ontology=config.nlp.use_scheme_ontology,
            limit=config.cbr.limit,
            mac_phase=config.cbr.mac,
            fac_phase=config.cbr.fac,
            mapping_algorithm=retrieval_pb2.MappingAlgorithm.Value(
                config.cbr.mapping_algorithm
            ),
            nlp_config=_nlp_configs[config.nlp.config],
        )

        if isinstance(query, ag.Graph):
            req.query_graph.CopyFrom(query.to_protobuf())
        elif isinstance(query, str):
            req.query_text = query
        else:
            raise ValueError(
                f"Query '{query_file}' has the unsupported type '{type(query).__name__}'"
            )

        res: retrieval_pb2.RetrieveResponse = client.Retrieve(req)

        evaluation = None
        mac_export = None
        fac_export = None

        if mac_results := res.mac_ranking:
            mac_export = exporter.get_results(protobuf_cases, mac_results)
            evaluation = Evaluation(cases, mac_results, query_file)

        if res.fac_ranking:
            fac_results = [result.case for result in res.fac_ranking]
            fac_export = exporter.get_results(protobuf_cases, fac_results)
            evaluation = Evaluation(cases, fac_results, query_file)

        evaluations.append(evaluation)

        if config.export.individual_results:
            exporter.export_results(
                query_file,
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
