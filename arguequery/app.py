import typing as t

import arg_services
import arguebuf as ag
import grpc
import typer
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc

from arguequery.models.graph import from_protobuf
from arguequery.services import retrieval
from arguequery.services.nlp import Nlp
from arguequery.types import RetrieveRequestMeta

_T = t.TypeVar("_T")


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def __init__(self, nlp_address: str) -> None:
        self.nlp_address = nlp_address

    def Retrieve(self, req: retrieval_pb2.RetrieveRequest, ctx: grpc.ServicerContext):
        responses: list[retrieval_pb2.QueryResponse] = []

        try:
            nlp = Nlp(self.nlp_address, req.nlp_config, req.scheme_handling)
            cases = {key: from_protobuf(value) for key, value in req.cases.items()}

            for query in req.queries:
                mac_similarities = {}
                fac_similarities = {}
                fac_mappings = {}

                query = from_protobuf(query)

                if req.semantic_retrieval:
                    mac_similarities = retrieval.mac(cases, query, nlp)

                filtered_mac_similarities = _filter(_sort(mac_similarities), req.limit)

                if req.structural_retrieval:
                    fac_cases = (
                        {key: cases[key] for key, _ in filtered_mac_similarities}
                        if filtered_mac_similarities
                        else cases
                    )
                    fac_results = retrieval.fac(
                        fac_cases,
                        query,
                        req.mapping_algorithm,
                        nlp,
                        req.extras["astar_queue_limit"],
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
        except Exception as e:
            arg_services.handle_except(e, ctx)

        return retrieval_pb2.RetrieveResponse(query_responses=responses)


def _sort(results: t.Mapping[str, float]) -> t.List[t.Tuple[str, float]]:
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def _filter(results: t.Sequence[_T], limit: int) -> t.Sequence[_T]:
    return results[:limit] if limit else results


app = typer.Typer()


def _get_serve_callback(nlp_address: str):
    def callback(server: grpc.Server):
        retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
            RetrievalService(nlp_address), server
        )

    return callback


@app.command()
def main(
    retrieval_address: str = "127.0.0.1:50055",
    nlp_address: str = "127.0.0.1:50051",
):
    """Main entry point for the server."""

    arg_services.serve(
        retrieval_address,
        _get_serve_callback(nlp_address),
        [arg_services.full_service_name(retrieval_pb2, "RetrievalService")],
    )
