import typing as t

import arg_services_helper
import arguebuf as ag
import grpc
from arg_services.retrieval.v1 import retrieval_pb2, retrieval_pb2_grpc

from arguequery.services import nlp, retrieval


class RetrievalService(retrieval_pb2_grpc.RetrievalServiceServicer):
    def Retrieve(self, req: retrieval_pb2.RetrieveRequest, ctx: grpc.ServicerContext):
        try:
            res = retrieval_pb2.RetrieveResponse()
            mac_results = {}
            fac_results = {}
            cases = {
                key: ag.Graph.from_protobuf(value) for key, value in req.cases.items()
            }

            # WARNING: The server currently is NOT thread safe due to the way the NLP config is handled
            nlp._nlp_config = req.nlp_config
            nlp._vector_cache = {}

            if (
                req.retrieval_method
                in [
                    retrieval_pb2.RetrievalMethod.RETRIEVAL_METHOD_MAC,
                    retrieval_pb2.RetrievalMethod.RETRIEVAL_METHOD_MAC_FAC,
                ]
                or req.WhichOneof("query") == "query_text"
            ):
                query = (
                    str(req.query_text)
                    if req.WhichOneof("query") == "query_text"
                    else ag.Graph.from_protobuf(req.query_graph)
                )
                mac_results = retrieval.mac(cases, query)

            if (
                req.retrieval_method
                in [
                    retrieval_pb2.RetrievalMethod.RETRIEVAL_METHOD_FAC,
                    retrieval_pb2.RetrievalMethod.RETRIEVAL_METHOD_MAC_FAC,
                ]
                and req.WhichOneof("query") == "query_graph"
            ):
                query = ag.Graph.from_protobuf(req.query_graph)

                filtered_mac_results = _filter(_sort(mac_results), req.limit)
                fac_cases = {key: cases[key] for key, _ in filtered_mac_results}

                fac_results = retrieval.fac(fac_cases, query)

            results = fac_results or mac_results
            filtered_results = _filter(_sort(results), req.limit)

            for key, sim in filtered_results:
                retrieved_case = retrieval_pb2.RetrievedCase(
                    case_id=key, similarity=sim
                )

                if mac_sim := mac_results.get(key):
                    retrieved_case.mac_similarity = mac_sim

                if fac_sim := fac_results.get(key):
                    retrieved_case.fac_similarity = fac_sim

                res.cases.append(retrieved_case)

            return res

        except Exception as e:
            arg_services_helper.handle_except(e, ctx)


def _sort(results: t.Mapping[str, float]) -> t.List[t.Tuple[str, float]]:
    return sorted(results.items(), key=lambda x: x[1])


def _filter(
    results: t.Sequence[t.Tuple[str, float]], limit: int
) -> t.Sequence[t.Tuple[str, float]]:
    return results[:limit] if limit else results
