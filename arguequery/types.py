import enum
import typing as t
from dataclasses import dataclass

from mashumaro.mixins.dict import DataClassDictMixin


class MappingAlgorithm(enum.Enum):
    ASTAR_1 = "astar_1"
    ASTAR_2 = "astar_2"
    ASTAR_3 = "astar_3"
    GREEDY_1 = "greedy_1"
    GREEDY_2 = "greedy_2"
    ISOMORPHISM_1 = "isomorphism_1"


class Graph2TextAlgorithm(enum.Enum):
    DFS = "dfs"
    DFS_RECONSTRUCTION = "dfs_reconstruction"
    BFS = "bfs"
    RANDOM = "random"
    ORIGINAL_RESOURCE = "original_resource"
    NODE_ID = "node_id"


@dataclass
class RetrieveRequestMeta(DataClassDictMixin):
    mapping_algorithm: MappingAlgorithm
    use_scheme_ontology: bool
    enforce_scheme_types: bool
    case_texts: dict[str, str]
    query_text: str
