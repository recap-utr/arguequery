from __future__ import absolute_import, annotations

import csv
import json
import logging
import time
import typing as t
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arguebuf as ag
import numpy as np
from arg_services.retrieval.v1 import retrieval_pb2
from arguequery.models.result import Result
from arguequery.services.evaluation import Evaluation

logger = logging.getLogger("recap")
from arguequery.config import config


def get_results(
    cases: t.Mapping[str, ag.Graph], results: t.Sequence[retrieval_pb2.RetrievedCase]
) -> List[Dict[str, Any]]:
    """Convert the results to strings"""

    return [
        {
            "name": cases[result.id].name,
            "rank": i + 1,
            "similarity": np.around(result.similarity, 3),
            "text": cases[result.id].name,  # TODO
        }
        for i, result in enumerate(results)
    ]


def export_results(
    query_file_name: str,
    mac_results: Optional[List[Dict[str, Any]]],
    fac_results: Optional[List[Dict[str, Any]]],
    evaluation: Optional[Evaluation],
) -> None:
    """Write the results to csv files

    The files will have mac, fac and eval appended to differentiate.
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    results_path = Path(config.path.results)
    results_path.mkdir(parents=True, exist_ok=True)

    folder = results_path / timestamp / query_file_name
    fieldnames = ["name", "rank", "similarity", "text"]

    if mac_results:
        with (folder / "mac.csv").open("w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows(mac_results)

    if fac_results:
        with (folder / "fac.csv").open("w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames)
            csvwriter.writeheader()
            csvwriter.writerows(fac_results)

    if evaluation:
        eval_dict = evaluation.as_dict()
        with (folder / "eval.csv").open("w", newline="") as csvfile:
            csvwriter = csv.DictWriter(csvfile, ["metric", "value"])
            csvwriter.writeheader()

            if "unranked" in eval_dict:
                for key, value in eval_dict["unranked"].items():
                    csvwriter.writerow({"metric": key, "value": value})

            if "ranked" in eval_dict:
                for key, value in eval_dict["ranked"].items():
                    csvwriter.writerow({"metric": key, "value": value})


def get_results_aggregated(
    evaluations: t.Collection[t.Optional[Evaluation]],
) -> Dict[str, t.DefaultDict[str, float]]:
    """Return multiple evaluations as an aggregated dictionary."""

    ranked_aggr: Dict[str, float] = defaultdict(float)
    unranked_aggr: Dict[str, float] = defaultdict(float)

    for evaluation in evaluations:
        if evaluation:
            eval_dict = evaluation.as_dict()

            if "unranked" in eval_dict:
                for key, value in eval_dict["unranked"].items():
                    unranked_aggr[key] += value

            if "ranked" in eval_dict:
                for key, value in eval_dict["ranked"].items():
                    ranked_aggr[key] += value

    eval_dict_aggr = {"unranked": unranked_aggr, "ranked": ranked_aggr}

    for eval_type in eval_dict_aggr.values():
        for key, value in eval_type.items():
            eval_type[key] = round((value) / len(evaluations), 3)

    return eval_dict_aggr


def export_results_aggregated(
    evaluation: t.Mapping[str, t.Mapping[str, float]],
    duration: float,
    parameters: t.Mapping[str, t.Any],
) -> None:
    """Write the results to file"""

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    results_path = Path(config.path.results)
    results_path.mkdir(parents=True, exist_ok=True)

    file = (results_path / timestamp).with_suffix(".json")

    with file.open("w") as f:
        tex_values = []

        for eval_type in evaluation.values():
            for value in eval_type.values():
                tex_values.append(r"\(" + str(round(value, 3)) + r"\)")

        json_out = {
            "Results": evaluation,
            "Duration": round(duration, 3),
            "Parameters": parameters,
        }
        json.dump(json_out, f, indent=4, ensure_ascii=False)
