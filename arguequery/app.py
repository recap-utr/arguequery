from __future__ import annotations

import logging
import typing as t
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import arguebuf as ag

from arguequery.models.result import Result
from arguequery.services import exporter, nlp, retrieval
from arguequery.services.evaluation import Evaluation

log = logging.getLogger("recap")
log.setLevel(logging.INFO)

root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)

from arguequery.config import config


def run() -> None:
    """Calculate similarity of queries and case base"""

    mac_results = None
    fac_results = None
    start_time = 0
    duration = 0
    eval_dict = {}

    if config["perform_mac"] or config["perform_fac"]:
        graphs = {
            file.name: ag.Graph.from_file(file)
            for file in Path(config.casebase_folder).glob("*.json")
        }
        query_graphs = {
            file.name: ag.Graph.from_file(file)
            for file in Path(config.queries_folder).glob("*.json")
        }

        start_time = timer()

        evaluations: List[t.Optional[Evaluation]] = []

        for number_of_run in range(1, config["number_of_runs"] + 1):
            log.info(f"Run {number_of_run} of {config['number_of_runs']}")

            for query_graph in query_graphs.values():
                mac = [Result(graph, 0.0) for graph in graphs.values()]
                fac = None
                evaluation = None

                if config["perform_mac"]:
                    mac = [
                        Result(graph, nlp.similarity(graph, query_graph))
                        for graph in graphs.values()
                    ]
                    mac_results = exporter.get_results(mac)

                    if config["retrieval_limit"] > 0:
                        mac = mac[: config["retrieval_limit"]]

                    evaluation = Evaluation(graphs, mac, query_graph)

                if config["perform_fac"]:
                    fac = retrieval.fac(mac, query_graph)
                    fac_results = exporter.get_results(fac)

                    if config["retrieval_limit"] > 0:
                        fac = fac[: config["retrieval_limit"]]

                    evaluation = Evaluation(graphs, fac, query_graph)

                evaluations.append(evaluation)

                if config["export_results"]:
                    exporter.export_results(
                        query_graph.name,
                        mac_results,
                        fac_results,
                        evaluation,
                    )
                    log.info("Individual Results were exported.")

        duration = (timer() - start_time) / config["number_of_runs"]
        eval_dict = exporter.get_results_aggregated(evaluations)

        if config["export_results_aggregated"]:
            exporter.export_results_aggregated(eval_dict, duration, **config)
            log.info("Aggregated Results were exported.")

        if len(query_graphs) > 1:
            mac_results = ""
            fac_results = ""
