from __future__ import annotations

import json
import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from typing import List

import arguebuf as ag
import grpc
import typer
from arg_services.nlp.v1 import nlp_pb2
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc
from rich import print, print_json

from arguequery.algorithms.graph2text import graph2text
from arguequery.config import config
from arguequery.services import exporter, nlp, retrieval
from arguequery.services.evaluation import Evaluation
from arguequery.types import RetrieveRequestMeta

log = logging.getLogger(__name__)
app = typer.Typer()

_nlp_configs = {
    "default": nlp_pb2.NlpConfig(
        language=config.client.request.language,
        spacy_model="en_core_web_lg",
        similarity_method=nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_COSINE,
    )
}


@app.command()
def main(retrieval_address: t.Optional[str] = None) -> None:
    """Calculate similarity of queries and case base"""

    start_time = 0
    duration = 0
    eval_dict = {}
    evaluations: List[t.Optional[Evaluation]] = []

    client = retrieval_pb2_grpc.RetrievalServiceStub(
        grpc.insecure_channel(retrieval_address or config.retrieval_address)
    )

    cases: t.Dict[Path, ag.Graph] = {
        file: ag.Graph.from_file(file)
        for file in Path(config.client.path.cases).glob(
            config.client.path.case_graphs_pattern
        )
    }
    arguebuf_cases = {
        str(key.relative_to(config.client.path.cases)): graph
        for key, graph in cases.items()
    }
    protobuf_cases = {key: graph.to_protobuf() for key, graph in arguebuf_cases.items()}

    queries: t.Dict[Path, t.Union[str, ag.Graph]] = {
        file: ag.Graph.from_file(file)
        for file in Path(config.client.path.queries).glob(
            config.client.path.query_graphs_pattern
        )
    }

    for file in Path(config.client.path.queries).glob(
        config.client.path.query_texts_pattern
    ):
        with file.open("r", encoding="utf-8") as f:
            queries[file] = f.read()

    start_time = timer()

    for query_file, query in queries.items():
        req = retrieval_pb2.RetrieveRequest(
            case_graphs=protobuf_cases,
            limit=config.client.request.limit,
            semantic_retrieval=config.client.request.mac,
            structural_retrieval=config.client.request.fac,
            nlp_config=_nlp_configs[config.client.request.nlp_config],
        )
        req.extras.update(
            RetrieveRequestMeta(
                mapping_algorithm=config.client.request.mapping_algorithm,
                use_scheme_ontology=config.client.request.use_scheme_ontology,
                enforce_scheme_types=config.client.request.enforce_scheme_types,
                query_text=query
                if isinstance(query, str)
                else graph2text(query, config.client.request.graph2text_algorithm),
                case_texts={
                    key: graph2text(graph, config.client.request.graph2text_algorithm)
                    for key, graph in arguebuf_cases.items()
                },
            ).to_dict()
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

        if mac_results := res.semantic_ranking:
            mac_export = exporter.get_results(mac_results)
            evaluation = Evaluation(cases, mac_results, query_file)

        if fac_results := res.structural_ranking:
            fac_export = exporter.get_results(fac_results)
            evaluation = Evaluation(cases, fac_results, query_file)

        evaluations.append(evaluation)

        if config.client.evaluation.individual_results:
            exporter.export_results(
                query_file,
                mac_export,
                fac_export,
                evaluation,
            )
            log.info("Individual Results were exported.")

    duration = timer() - start_time
    eval_dict = exporter.get_results_aggregated(evaluations)

    print_json(json.dumps(eval_dict))

    if config.client.evaluation.aggregated_results:
        exporter.export_results_aggregated(eval_dict, duration, config.as_dict())
        log.info("Aggregated Results were exported.")
