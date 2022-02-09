import typing as t

import arg_services_helper
import arguebuf as ag
import grpc
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.config import config
from arguequery.services import nlp, retrieval


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def Retrieve(self, req: retrieval_pb2.RetrieveRequest, ctx: grpc.ServicerContext):
        try:
            mac_results = {}
            fac_results = {}
            cases = {
                key: ag.Graph.from_protobuf(value) for key, value in req.cases.items()
            }

            # WARNING: The server currently is NOT thread safe due to the way the NLP config is handled
            nlp.nlp_config = req.nlp_config
            nlp.vector_cache = {}
            nlp.use_scheme_ontology = req.use_scheme_ontology
            nlp.enforce_scheme_types = req.enforce_scheme_types

            if req.mac_phase or req.WhichOneof("query") == "query_text":
                query = (
                    str(req.query_text)
                    if req.WhichOneof("query") == "query_text"
                    else ag.Graph.from_protobuf(req.query_graph)
                )
                mac_results = retrieval.mac(cases, query)

            filtered_mac_results = _filter(_sort(mac_results), req.limit)

            if req.fac_phase and req.WhichOneof("query") == "query_graph":
                query = ag.Graph.from_protobuf(req.query_graph)
                fac_cases = (
                    {key: cases[key] for key, _ in filtered_mac_results}
                    if filtered_mac_results
                    else cases
                )
                fac_results = retrieval.fac(fac_cases, query)

            filtered_fac_results = _filter(_sort(fac_results), req.limit)
            filtered_results = filtered_fac_results or filtered_mac_results

            return retrieval_pb2.RetrieveResponse(
                results=[
                    retrieval_pb2.RetrievedCase(id=key, similarity=sim)
                    for key, sim in filtered_results
                ],
                mac_results=[
                    retrieval_pb2.RetrievedCase(id=key, similarity=sim)
                    for key, sim in filtered_mac_results
                ],
                fac_results=[
                    retrieval_pb2.RetrievedCase(id=key, similarity=sim)
                    for key, sim in filtered_fac_results
                ],
            )

        except Exception as e:
            arg_services_helper.handle_except(e, ctx)


def _sort(results: t.Mapping[str, float]) -> t.List[t.Tuple[str, float]]:
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def _filter(
    results: t.Sequence[t.Tuple[str, float]], limit: int
) -> t.Sequence[t.Tuple[str, float]]:
    return results[:limit] if limit else results


def _filter_names(results: t.Sequence[t.Tuple[str, float]], limit: int) -> t.List[str]:
    return [x[0] for x in _filter(results, limit)]


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(), server
    )


if __name__ == "__main__":
    host, port = config.retrieval_url.split(":")

    arg_services_helper.serve(
        host,
        port,
        add_services,
        reflection_services=[
            arg_services_helper.full_service_name(retrieval_pb2, "RetrievalService"),
        ],
    )
