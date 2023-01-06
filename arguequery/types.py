import typing as t
from dataclasses import dataclass

from mashumaro.mixins.dict import DataClassDictMixin

MappingAlgorithm = t.Literal["astar", "isomorphism"]


@dataclass
class RetrieveRequestMeta(DataClassDictMixin):
    mapping_algorithm: MappingAlgorithm
    use_scheme_ontology: bool
    enforce_scheme_types: bool
