import arg_services
import cbrkit
import grpc
from arg_services.cbr.v1beta import retrieval_pb2, retrieval_pb2_grpc
from typer import Typer

from .model import EdgeData, GraphData, KeyType, NodeData, load_graph
from .sim import Similarity


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def Similarities(
        self, request: retrieval_pb2.SimilaritiesRequest, context: grpc.ServicerContext
    ) -> retrieval_pb2.SimilaritiesResponse:
        similarity = Similarity(
            request.nlp_config,
            None,
            request.mapping_algorithm,
            request.mapping_algorithm_variant,
            request.scheme_handling,
            request.extras,
        )

        retriever = (
            similarity.retriever_fac if request.structural else similarity.retriever_mac
        )

        request_cases = {str(i): case for i, case in enumerate(request.cases)}
        cases = {key: load_graph(case) for key, case in request_cases.items()}
        query = load_graph(request.query)

        result = cbrkit.retrieval.apply_query(cases, query, retriever)

        return retrieval_pb2.SimilaritiesResponse(
            similarities=[
                retrieval_pb2.SimilarityResponse(
                    semantic_similarity=cbrkit.helpers.unpack_float(sim)
                    if not request.structural
                    else None,
                    structural_similarity=cbrkit.helpers.unpack_float(sim)
                    if request.structural
                    else None,
                    structural_mapping=[
                        retrieval_pb2.RetrievedMapping(
                            case=retrieval_pb2.RetrievedCase(
                                id=case_key,
                                similarity=cbrkit.helpers.unpack_float(sim),
                                graph=request_cases[case_key],
                            ),
                            node_mappings=[
                                retrieval_pb2.MappedElement(
                                    query_id=query_node,
                                    case_id=case_node,
                                    similarity=cbrkit.helpers.unpack_float(
                                        sim.node_similarities[query_node]
                                    ),
                                )
                            ],
                        )
                        for query_node, case_node in sim.node_mapping.items()
                    ]
                    if isinstance(sim, cbrkit.sim.graphs.GraphSim)
                    else [],
                )
                for case_key, sim in result.similarities.items()
            ]
        )

    def Retrieve(
        self, request: retrieval_pb2.RetrieveRequest, context: grpc.ServicerContext
    ) -> retrieval_pb2.RetrieveResponse:
        try:
            retrievers: cbrkit.typing.MaybeFactories[
                cbrkit.typing.RetrieverFunc[
                    KeyType,
                    cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
                    float | cbrkit.sim.graphs.GraphSim[KeyType],
                ]
            ] = []
            similarity = Similarity(
                request.nlp_config,
                request.limit,
                request.mapping_algorithm,
                request.mapping_algorithm_variant,
                request.scheme_handling,
                request.extras,
            )

            if request.semantic_retrieval:
                retrievers.append(similarity.retriever_mac)

            if request.structural_retrieval:
                retrievers.append(similarity.retriever_fac_precompute)
                retrievers.append(similarity.retriever_fac)

            queries = {key: load_graph(query) for key, query in request.queries.items()}
            cases = {key: load_graph(value) for key, value in request.cases.items()}

            result = cbrkit.retrieval.apply_queries(cases, queries, retrievers)

            return retrieval_pb2.RetrieveResponse(
                query_responses={
                    query_key: retrieval_pb2.QueryResponse(
                        semantic_ranking=[
                            retrieval_pb2.RetrievedCase(
                                id=case_key,
                                similarity=cbrkit.helpers.unpack_float(sim),
                                graph=request.cases[case_key],
                            )
                            for case_key, sim in result.steps[0]
                            .queries[query_key]
                            .similarities.items()
                        ]
                        if request.semantic_retrieval
                        else [],
                        structural_ranking=[
                            retrieval_pb2.RetrievedCase(
                                id=case_key,
                                similarity=cbrkit.helpers.unpack_float(sim),
                                graph=request.cases[case_key],
                            )
                            for case_key, sim in result.steps[-1]
                            .queries[query_key]
                            .similarities.items()
                        ]
                        if request.structural_retrieval
                        else [],
                        structural_mapping=[
                            retrieval_pb2.RetrievedMapping(
                                case=retrieval_pb2.RetrievedCase(
                                    id=case_key,
                                    similarity=cbrkit.helpers.unpack_float(sim),
                                    graph=request.cases[case_key],
                                ),
                                node_mappings=[
                                    retrieval_pb2.MappedElement(
                                        query_id=query_node,
                                        case_id=case_node,
                                        similarity=cbrkit.helpers.unpack_float(
                                            sim.node_similarities[query_node]
                                        ),
                                    )
                                    for query_node, case_node in sim.node_mapping.items()
                                ]
                                if isinstance(sim, cbrkit.sim.graphs.GraphSim)
                                else [],
                            )
                            for case_key, sim in result.steps[-1]
                            .queries[query_key]
                            .similarities.items()
                        ]
                        if len(result.steps) == 2
                        else [],
                    )
                    for query_key in request.queries.keys()
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            raise e


def add_services(server: grpc.Server):
    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(
        RetrievalService(), server
    )


app = Typer()


@app.command()
def main(address: str = "localhost:50200"):
    """Main entry point for the server."""

    arg_services.serve(
        address,
        add_services,
        [arg_services.full_service_name(retrieval_pb2, "RetrievalService")],
    )
