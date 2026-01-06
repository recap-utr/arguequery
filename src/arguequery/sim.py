import functools
from dataclasses import dataclass
from pathlib import Path
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

nlp_with_models = nlp_service.Nlp(
    cache_path=Path("data/embeddings.sqlite3"),
)
nlp_without_models = nlp_service.Nlp(
    cache_path=Path("data/embeddings.sqlite3"),
    provider_init=None,
    provider_cache=False,
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
    config: nlp_service.model.NlpConfig
    limit: int | None
    mapping_algorithm: retrieval_pb2.MappingAlgorithm
    mapping_algorithm_variant: int
    scheme_handling: retrieval_pb2.SchemeHandling
    extras: Struct
    multiprocessing: bool

    def node_matcher(self, x: NodeData, y: NodeData) -> bool:
        return type(x) is type(y)

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
        cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        return cbrkit.sim.transpose_value(nlp_with_models.sim_func(self.config))

    @property
    def graph_fac(
        self,
    ) -> cbrkit.typing.AnySimFunc[
        cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
        cbrkit.sim.graphs.GraphSim[KeyType],
    ]:
        node_sim_func = cbrkit.sim.type_table(
            {
                str: nlp_without_models.sim_func(self.config),
                SchemeData: self.scheme,
            },
        )

        match self.mapping_algorithm:
            case retrieval_pb2.MAPPING_ALGORITHM_ASTAR:
                astar_func = functools.partial(
                    cbrkit.sim.graphs.astar.build,
                    node_sim_func=node_sim_func,
                    node_matcher=self.node_matcher,
                    beam_width=(
                        0
                        if "astar_beam_width" not in self.extras
                        else int(cast(float, self.extras["astar_beam_width"]))
                    ),
                    pathlength_weight=(
                        0
                        if "astar_pathlength_weight" not in self.extras
                        else int(cast(float, self.extras["astar_pathlength_weight"]))
                    ),
                )

                match self.mapping_algorithm_variant:
                    case 1:
                        return astar_func(
                            heuristic_func=cbrkit.sim.graphs.astar.h1(),
                            selection_func=cbrkit.sim.graphs.astar.select1(),
                            init_func=cbrkit.sim.graphs.init_empty(),
                        )
                    case 2:
                        return astar_func(
                            heuristic_func=cbrkit.sim.graphs.astar.h2(),
                            selection_func=cbrkit.sim.graphs.astar.select2(),
                            init_func=cbrkit.sim.graphs.init_empty(),
                        )
                    case 3:
                        return astar_func(
                            heuristic_func=cbrkit.sim.graphs.astar.h3(),
                            selection_func=cbrkit.sim.graphs.astar.select3(),
                            init_func=cbrkit.sim.graphs.init_unique_matches(),
                        )
                    case 4:
                        return astar_func(
                            heuristic_func=cbrkit.sim.graphs.astar.h4(),
                            selection_func=cbrkit.sim.graphs.astar.select4(),
                            init_func=cbrkit.sim.graphs.init_unique_matches(),
                        )

                raise ValueError(
                    f"Unknown mapping_algorithm_variant: {self.mapping_algorithm_variant}"
                )
            case retrieval_pb2.MAPPING_ALGORITHM_BRUTE_FORCE:
                return cbrkit.sim.graphs.brute_force(
                    node_sim_func,
                    node_matcher=self.node_matcher,
                )
            case retrieval_pb2.MAPPING_ALGORITHM_VF2:
                return cbrkit.sim.graphs.vf2(
                    node_sim_func,
                    node_matcher=self.node_matcher,
                )
            case retrieval_pb2.MAPPING_ALGORITHM_DFS:
                return cbrkit.sim.graphs.dfs(
                    node_sim_func,
                    node_matcher=self.node_matcher,
                )
            case retrieval_pb2.MAPPING_ALGORITHM_GREEDY:
                return cbrkit.sim.graphs.greedy(
                    node_sim_func,
                    node_matcher=self.node_matcher,
                )
            case retrieval_pb2.MAPPING_ALGORITHM_LSAP:
                return cbrkit.sim.graphs.lap(
                    node_sim_func,
                    node_matcher=self.node_matcher,
                )
            case _:
                raise ValueError(f"Unknown mapping_algorithm: {self.mapping_algorithm}")

    def retriever_mac(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        retriever = cbrkit.retrieval.transpose_value(
            nlp_with_models.retrieval_func(self.config)
        )

        if self.limit is not None:
            return cbrkit.retrieval.dropout(retriever, limit=self.limit)

        return retriever

    def retriever_fac(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
        cbrkit.sim.graphs.GraphSim[KeyType],
    ]:
        retriever = cbrkit.retrieval.build(
            self.graph_fac,
            multiprocessing=self.multiprocessing,
            chunksize=1 if self.multiprocessing else 0,
        )

        if self.limit is not None:
            return cbrkit.retrieval.dropout(retriever, limit=self.limit)

        return retriever

    def retriever_fac_precompute(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.model.graph.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        return cbrkit.retrieval.build(
            cbrkit.sim.graphs.precompute(
                node_sim_func=cbrkit.sim.type_table(
                    {str: nlp_with_models.sim_func(self.config)},
                    default=0.0,
                ),
                node_matcher=self.node_matcher,
            )
        )
