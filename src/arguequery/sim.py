import functools
from dataclasses import dataclass
from typing import cast

import arguebuf
import cbrkit
import nlp_service
from arg_services.cbr.v1beta import retrieval_pb2
from google.protobuf.struct_pb2 import Struct

from arguequery.model import (
    EdgeData,
    GraphData,
    KeyType,
    NodeData,
    SchemeData,
)


@functools.cache
def _scheme2value(node: SchemeData) -> arguebuf.Scheme | None:
    return node.scheme


@functools.cache
def _scheme2str(node: SchemeData) -> str:
    if node.scheme is None:
        return ""

    return node.scheme.name.lower().replace("_", " ")


@dataclass(frozen=True, slots=True)
class Similarity:
    nlp: nlp_service.Nlp
    mapping_algorithm: retrieval_pb2.MappingAlgorithm
    mapping_algorithm_variant: int
    scheme_handling: retrieval_pb2.SchemeHandling
    extras: Struct

    @property
    def scheme(self) -> cbrkit.typing.AnySimFunc[SchemeData, float]:
        match self.scheme_handling:
            case retrieval_pb2.SCHEME_HANDLING_UNSPECIFIED:
                return cbrkit.sim.generic.static(1.0)
            case retrieval_pb2.SCHEME_HANDLING_BINARY:
                return cbrkit.sim.transpose(
                    cbrkit.sim.generic.type_equality(),
                    _scheme2value,
                )
            case retrieval_pb2.SCHEME_HANDLING_TAXONOMY:
                return cbrkit.sim.transpose(
                    cbrkit.sim.taxonomy.build(
                        "schemes.yml",
                        cbrkit.sim.taxonomy.wu_palmer(),
                    ),
                    _scheme2str,
                )
            case retrieval_pb2.SCHEME_HANDLING_EXACT:
                return cbrkit.sim.transpose(
                    cbrkit.sim.generic.equality(),
                    _scheme2value,
                )

        raise ValueError(f"Unknown scheme_handling: {self.scheme_handling}")

    @property
    def graph_mac(
        self,
    ) -> cbrkit.typing.AnySimFunc[
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        return cbrkit.sim.transpose_value(self.nlp.similarity)

    @property
    def graph_fac(
        self,
    ) -> cbrkit.typing.AnySimFunc[
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        cbrkit.sim.graphs.GraphSim[KeyType],
    ]:
        node_sim_func = cbrkit.sim.transpose_value(
            cbrkit.sim.type_table(
                {
                    str: self.nlp.similarity,
                    SchemeData: self.scheme,
                },
                default=cbrkit.sim.generic.static(0.0),
            )
        )
        precompute_nodes_func = cbrkit.sim.transpose_value(
            cbrkit.sim.type_table(
                {str: self.nlp.similarity},
                default=cbrkit.sim.generic.static(0.0),
            )
        )

        match self.mapping_algorithm:
            case retrieval_pb2.MAPPING_ALGORITHM_ASTAR:
                queue_limit = 10000

                if "astar_queue_limit" in self.extras:
                    queue_limit = int(cast(float, self.extras["astar_queue_limit"]))

                match self.mapping_algorithm_variant:
                    case 1:
                        return cbrkit.sim.graphs.astar.build(
                            past_cost_func=cbrkit.sim.graphs.astar.g1(node_sim_func),
                            future_cost_func=cbrkit.sim.graphs.astar.h1(),
                            selection_func=cbrkit.sim.graphs.astar.select1(),
                            init_func=cbrkit.sim.graphs.astar.init1(),
                            queue_limit=queue_limit,
                            precompute_nodes_func=precompute_nodes_func,
                            multiprocessing=True,
                        )
                    case 2:
                        return cbrkit.sim.graphs.astar.build(
                            past_cost_func=cbrkit.sim.graphs.astar.g1(node_sim_func),
                            future_cost_func=cbrkit.sim.graphs.astar.h2(node_sim_func),
                            selection_func=cbrkit.sim.graphs.astar.select2(),
                            init_func=cbrkit.sim.graphs.astar.init1(),
                            queue_limit=queue_limit,
                            precompute_nodes_func=precompute_nodes_func,
                            multiprocessing=True,
                        )
                    case 3:
                        return cbrkit.sim.graphs.astar.build(
                            past_cost_func=cbrkit.sim.graphs.astar.g1(node_sim_func),
                            future_cost_func=cbrkit.sim.graphs.astar.h3(node_sim_func),
                            selection_func=cbrkit.sim.graphs.astar.select3(
                                cbrkit.sim.graphs.astar.h3(node_sim_func)
                            ),
                            init_func=cbrkit.sim.graphs.astar.init2(),
                            queue_limit=queue_limit,
                            precompute_nodes_func=precompute_nodes_func,
                            multiprocessing=True,
                        )

                raise ValueError(
                    f"Unknown mapping_algorithm_variant: {self.mapping_algorithm_variant}"
                )
            case retrieval_pb2.MAPPING_ALGORITHM_BRUTE_FORCE:
                return cbrkit.sim.graphs.brute_force(node_sim_func)
            case retrieval_pb2.MAPPING_ALGORITHM_ISOMORPHISM:
                return cbrkit.sim.graphs.isomorphism(node_sim_func=node_sim_func)
            case _:
                raise ValueError(f"Unknown mapping_algorithm: {self.mapping_algorithm}")

    @property
    def retriever_mac(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        return cbrkit.retrieval.transpose_value(self.nlp.retrieval)

    @property
    def retriever_fac(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        cbrkit.sim.graphs.GraphSim[KeyType],
    ]:
        return cbrkit.retrieval.build(self.graph_fac)
