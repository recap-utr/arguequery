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

# def sentence_transformer_init(
#     model_name: str,
# ) -> cbrkit.typing.BatchConversionFunc[str, cbrkit.typing.NumpyArray]:
#     # mps has issues with multiprocessing on macos
#     return cbrkit.sim.embed.sentence_transformers(
#         SentenceTransformer(model_name, device="cuda" if is_cuda_available() else "cpu")
#     )

nlp_with_models = nlp_service.Nlp(
    cache_dir=Path("data"),
    autodump=True,
)
nlp_without_models = nlp_service.Nlp(
    cache_dir=Path("data"),
    autodump=False,
    provider_init=None,
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
    config: nlp_service.NlpConfig
    limit: int | None
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
        return cbrkit.sim.transpose_value(nlp_with_models.sim_func(self.config))

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
                    str: nlp_without_models.sim_func(self.config),
                    SchemeData: self.scheme,
                },
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
                        )
                    case 2:
                        return cbrkit.sim.graphs.astar.build(
                            past_cost_func=cbrkit.sim.graphs.astar.g1(node_sim_func),
                            future_cost_func=cbrkit.sim.graphs.astar.h2(node_sim_func),
                            selection_func=cbrkit.sim.graphs.astar.select2(),
                            init_func=cbrkit.sim.graphs.astar.init1(),
                            queue_limit=queue_limit,
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
        retriever = cbrkit.retrieval.transpose_value(
            nlp_with_models.retrieval_func(self.config)
        )

        if self.limit is not None:
            return cbrkit.retrieval.dropout(retriever, limit=self.limit)

        return retriever

    # no property, this should be a factory that lazily loads the embedding cache
    def retriever_fac(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        cbrkit.sim.graphs.GraphSim[KeyType],
    ]:
        retriever = cbrkit.retrieval.build(
            self.graph_fac, multiprocessing=True, chunksize=1
        )

        if self.limit is not None:
            return cbrkit.retrieval.dropout(retriever, limit=self.limit)

        return retriever

    @property
    def retriever_fac_precompute(
        self,
    ) -> cbrkit.typing.RetrieverFunc[
        KeyType,
        cbrkit.sim.graphs.Graph[KeyType, NodeData, EdgeData, GraphData],
        float,
    ]:
        precompute_nodes_func = cbrkit.sim.transpose_value(
            cbrkit.sim.type_table(
                {str: nlp_with_models.sim_func(self.config)},
                default=cbrkit.sim.generic.static(0.0),
            )
        )

        return cbrkit.retrieval.build(
            cbrkit.sim.graphs.precompute(precompute_nodes_func)
        )
