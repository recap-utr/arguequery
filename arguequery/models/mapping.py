import typing as t
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class FacMapping:
    query_id: str
    case_id: str
    similarity: float


@dataclass
class FacResults:
    similarities: t.Dict[str, float]
    mappings: t.Dict[str, t.Set[FacMapping]]
