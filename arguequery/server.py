import typing as t

import arg_services
import arguebuf as ag
import grpc
import typer
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.config import config
from arguequery.services import nlp, retrieval
from arguequery.types import RetrieveRequestMeta


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def Retrieve(self, req: retrieval_pb2.RetrieveRequest, ctx: grpc.ServicerContext):
        try:
            mac_similarities = {}
            fac_similarities = {}
            fac_mappings = {}
            cases = {
                key: ag.Graph.from_protobuf(value)
                for key, value in req.case_graphs.items()
            }

            # WARNING: The server currently is NOT thread safe due to the way the NLP config is handled
            req_meta = RetrieveRequestMeta.from_dict(dict(req.extras.items()))
            nlp.nlp_config = req.nlp_config
            nlp.vector_cache = {}
            nlp.use_scheme_ontology = req_meta.use_scheme_ontology
            nlp.enforce_scheme_types = req_meta.enforce_scheme_types
            nlp.query_text = req_meta.query_text
            nlp.case_texts = req_meta.case_texts

            if req.semantic_retrieval or req.WhichOneof("query") == "query_text":
                query = (
                    str(req.query_text)
                    if req.WhichOneof("query") == "query_text"
                    else ag.Graph.from_protobuf(req.query_graph)
                )
                mac_similarities = retrieval.mac(cases, query)

            filtered_mac_similarities = _filter(_sort(mac_similarities), req.limit)

            if req.structural_retrieval and req.WhichOneof("query") == "query_graph":
                query = ag.Graph.from_protobuf(req.query_graph)
                fac_cases = (
                    {key: cases[key] for key, _ in filtered_mac_similarities}
                    if filtered_mac_similarities
                    else cases
                )
                fac_results = retrieval.fac(
                    fac_cases, query, req_meta.mapping_algorithm
                )
                fac_similarities = fac_results.similarities
                fac_mappings = fac_results.mappings

            filtered_fac_similarities = _filter(_sort(fac_similarities), req.limit)

            return retrieval_pb2.RetrieveResponse(
                semantic_ranking=[
                    retrieval_pb2.RetrievedCase(id=key, similarity=sim)
                    for key, sim in filtered_mac_similarities
                ],
                structural_ranking=[
                    retrieval_pb2.RetrievedCase(id=key, similarity=sim)
                    for key, sim in filtered_fac_similarities
                ],
                structural_mapping=[
                    retrieval_pb2.RetrievedMapping(
                        case=retrieval_pb2.RetrievedCase(id=key, similarity=sim),
                        node_mappings=[
                            retrieval_pb2.Mapping(
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

        except Exception as e:
            arg_services.handle_except(e, ctx)


def _sort(results: t.Mapping[str, float]) -> t.List[t.Tuple[str, float]]:
    return sorted(results.items(), key=lambda x: x[1], reverse=True)


def _filter(
    results: t.Sequence[t.Tuple[str, float]], limit: int
) -> t.Sequence[t.Tuple[str, float]]:
    return results[:limit] if limit else results


def _filter_names(results: t.Sequence[t.Tuple[str, float]], limit: int) -> t.List[str]:
    return [x[0] for x in _filter(results, limit)]


app = typer.Typer()


def add_services(server: grpc.Server):
    """Add the services to the grpc server."""

    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(), server
    )


@app.command()
def main(
    retrieval_address: t.Optional[str] = None, nlp_address: t.Optional[str] = None
):
    """Main entry point for the server."""

    if nlp_address:
        nlp.address = nlp_address
        nlp.client = nlp.init_client()

    arg_services.serve(
        retrieval_address or config.retrieval_address,
        add_services,
        [arg_services.full_service_name(retrieval_pb2, "RetrievalService")],
    )


if __name__ == "__main__":
    app()
