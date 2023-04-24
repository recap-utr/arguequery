import logging
import typing as t

import arg_services
import grpc
import rich_click as click
import typed_settings as ts
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc

from arguequery.config import Config
from arguequery.models.graph import load
from arguequery.services import retrieval
from arguequery.services.nlp import Nlp

_T = t.TypeVar("_T")

log = logging.getLogger(__name__)


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def __init__(self, config: Config) -> None:
        self.config = config

    def Similarities(
        self, req: retrieval_pb2.SimilaritiesRequest, ctx: grpc.ServicerContext
    ) -> retrieval_pb2.SimilaritiesResponse:
        nlp = Nlp(self.config.nlp_address, req.nlp_config, req.scheme_handling)
        semantic_similarities = nlp.similarities(
            (case.text, req.query.text) for case in req.cases
        )

        return retrieval_pb2.SimilaritiesResponse(
            similarities=[
                retrieval_pb2.SimilarityResponse(semantic_similarity=sem_sim)
                for sem_sim in semantic_similarities
            ]
        )

    def Retrieve(self, req: retrieval_pb2.RetrieveRequest, ctx: grpc.ServicerContext):
        log.info(f"[{id(self)}] Processing request...")
        responses: list[retrieval_pb2.QueryResponse] = []
        nlp = Nlp(self.config.nlp_address, req.nlp_config, req.scheme_handling)
        cases = {key: load(value) for key, value in req.cases.items()}

        for i, query in enumerate(req.queries):
            log.debug(f"[{id(self)}] Processing query {i + 1}/{len(req.queries)}...")
            mac_similarities = {}
            fac_similarities = {}
            fac_mappings = {}

            query = load(query)

            if req.semantic_retrieval:
                mac_similarities = retrieval.mac(cases, query, nlp)

            filtered_mac_similarities = _filter(_sort(mac_similarities), req.limit)

            if req.structural_retrieval:
                fac_cases = (
                    {key: cases[key] for key, _ in filtered_mac_similarities}
                    if filtered_mac_similarities
                    else cases
                )

                try:
                    astar_queue_limit = int(req.extras["astar_queue_limit"])  # type: ignore
                except Exception:
                    astar_queue_limit = 10000

                fac_results = retrieval.fac(
                    fac_cases,
                    query,
                    req.mapping_algorithm,
                    nlp,
                    astar_queue_limit,
                )
                fac_similarities = fac_results.similarities
                fac_mappings = fac_results.mappings

            filtered_fac_similarities = _filter(_sort(fac_similarities), req.limit)

            responses.append(
                retrieval_pb2.QueryResponse(
                    semantic_ranking=[
                        retrieval_pb2.RetrievedCase(
                            id=key, similarity=sim, graph=req.cases[key]
                        )
                        for key, sim in filtered_mac_similarities
                    ],
                    structural_ranking=[
                        retrieval_pb2.RetrievedCase(
                            id=key, similarity=sim, graph=req.cases[key]
                        )
                        for key, sim in filtered_fac_similarities
                    ],
                    structural_mapping=[
                        retrieval_pb2.RetrievedMapping(
                            case=retrieval_pb2.RetrievedCase(
                                id=key, similarity=sim, graph=req.cases[key]
                            ),
                            node_mappings=[
                                retrieval_pb2.MappedElement(
                                    query_id=mapping.query_id,
                                    case_id=mapping.case_id,
                                    similarity=mapping.similarity,
                                )
                                for mapping in fac_mappings[key]
                            ],
                        )
                        for key, sim in filtered_fac_similarities
                    ],
                )
            )

        return retrieval_pb2.RetrieveResponse(query_responses=responses)


def _sort(results: t.Mapping[str, float]) -> t.List[t.Tuple[str, float]]:
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def _filter(results: t.Sequence[_T], limit: int) -> t.Sequence[_T]:
    return results[:limit] if limit else results


class ServiceAdder:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, server: grpc.Server):
        retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
            RetrievalService(self.config), server
        )


@click.command("arguequery")
@ts.click_options(Config, "config")
def main(
    config: Config,
):
    """Main entry point for the server."""

    arg_services.serve(
        config.address,
        ServiceAdder(config),
        [arg_services.full_service_name(retrieval_pb2, "RetrievalService")],
    )
