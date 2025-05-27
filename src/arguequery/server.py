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
        response = retrieval_pb2.SimilaritiesResponse()

        for case_key in result.ranking:
            sim = result.similarities[case_key]
            response.similarities.append(
                retrieval_pb2.SimilarityResponse(
                    similarity=cbrkit.helpers.unpack_float(sim),
                    mapping=retrieval_pb2.RetrievedMapping(
                        id=case_key,
                        node_mappings=[
                            retrieval_pb2.MappedElement(
                                query_id=query_node,
                                case_id=case_node,
                                similarity=cbrkit.helpers.unpack_float(
                                    sim.node_similarities[query_node]
                                ),
                            )
                            for query_node, case_node in sim.node_mapping.items()
                        ],
                        edge_mappings=[
                            retrieval_pb2.MappedElement(
                                query_id=query_edge,
                                case_id=case_edge,
                                similarity=cbrkit.helpers.unpack_float(
                                    sim.edge_similarities[query_edge]
                                ),
                            )
                            for query_edge, case_edge in sim.edge_mapping.items()
                        ],
                    )
                    if isinstance(sim, cbrkit.sim.graphs.GraphSim)
                    else None,
                )
            )

        return response

    def Retrieve(
        self, request: retrieval_pb2.RetrieveRequest, context: grpc.ServicerContext
    ) -> retrieval_pb2.RetrieveResponse:
        try:
            retrievers: cbrkit.typing.MaybeFactories[
                cbrkit.typing.RetrieverFunc[
                    KeyType,
                    cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
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
            response = retrieval_pb2.RetrieveResponse()

            for query_key in request.queries.keys():
                query_response = response.query_responses[query_key]
                semantic_result = (
                    result.steps[0].queries[query_key]
                    if request.semantic_retrieval
                    else None
                )
                structural_result = (
                    result.steps[-1].queries[query_key]
                    if request.structural_retrieval
                    else None
                )

                if semantic_result:
                    query_response.semantic_ranking.extend(
                        retrieval_pb2.RetrievedCase(
                            id=case_key,
                            similarity=cbrkit.helpers.unpack_float(
                                semantic_result.similarities[case_key]
                            ),
                            graph=request.cases[case_key],
                        )
                        for case_key in semantic_result.ranking
                    )

                if structural_result:
                    query_response.structural_ranking.extend(
                        retrieval_pb2.RetrievedCase(
                            id=case_key,
                            similarity=cbrkit.helpers.unpack_float(
                                structural_result.similarities[case_key]
                            ),
                            graph=request.cases[case_key],
                        )
                        for case_key in structural_result.ranking
                    )

                    query_response.structural_mapping.extend(
                        retrieval_pb2.RetrievedMapping(
                            id=case_key,
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
                            if (sim := structural_result.similarities[case_key])
                            and isinstance(sim, cbrkit.sim.graphs.GraphSim)
                            else [],
                            edge_mappings=[
                                retrieval_pb2.MappedElement(
                                    query_id=query_edge,
                                    case_id=case_edge,
                                    similarity=cbrkit.helpers.unpack_float(
                                        sim.edge_similarities[query_edge]
                                    ),
                                )
                                for query_edge, case_edge in sim.edge_mapping.items()
                            ]
                            if (sim := structural_result.similarities[case_key])
                            and isinstance(sim, cbrkit.sim.graphs.GraphSim)
                            else [],
                        )
                        for case_key in structural_result.ranking
                    )

            return response
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
