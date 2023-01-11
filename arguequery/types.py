import typing as t
from dataclasses import dataclass

from mashumaro.mixins.dict import DataClassDictMixin

MappingAlgorithm = t.Literal["astar", "isomorphism"]

Graph2TextAlgorithm = t.Literal[
    "dfs", "dfs_reconstruction", "bfs", "random", "original_resource", "node_id"
]


@dataclass
class RetrieveRequestMeta(DataClassDictMixin):
    mapping_algorithm: MappingAlgorithm
    use_scheme_ontology: bool
    enforce_scheme_types: bool
    case_texts: dict[str, str]
    query_text: str
