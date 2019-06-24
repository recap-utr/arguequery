from __future__ import absolute_import, annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set

import numpy as np

from ..services import utils

logger = logging.getLogger("recap")


@dataclass
class Node(object):
    """Class for storing node objects in graph contexts"""

    id_: int
    text: str
    type_: str
    vector: np.ndarray = field(init=False)
    vectors: List[np.ndarray] = field(init=False)
    tokens: List[str] = field(init=False)


@dataclass
class Edge(object):
    """Class for storing edge objects in graph contexts

    The node id's are automatically fetched from the node objects that are given.
    """

    id_: int
    from_node: Node
    to_node: Node
    from_id: int = field(init=False)
    to_id: int = field(init=False)

    def __post_init__(self) -> None:
        self.from_id = self.from_node.id_
        self.to_id = self.to_node.id_


@dataclass
class Graph(object):
    """Class for storing a graph by saving its nodes and edges"""

    all_nodes: Dict[int, Node]
    i_nodes: Dict[int, Node]
    s_nodes: Dict[int, Node]
    edges: Dict[int, Edge]
    filename: str
    text: str = field(init=False)
    vector: np.ndarray = field(init=False)
    vectors: List[np.ndarray] = field(init=False)
    tokens: List[str] = field(init=False)

    def __post_init__(self) -> None:
        self.text = get_text_from_nodes(self.i_nodes)


def get_text_from_nodes(nodes: Dict[int, Node]) -> str:
    """Concatenate text from multiple nodes to one string"""

    text_concat = ""

    for node in nodes.values():
        text_concat += node.text + " "

    return text_concat.strip()
