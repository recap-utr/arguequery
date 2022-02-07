from __future__ import absolute_import, annotations

import math
from collections import defaultdict
from typing import Dict, List, Set

from ..libs.sif import SifWeights
from ..models.graph import Graph
from ..services import utils

from recap_agr.config import config


class TokenWeighter(object):
    """Perform an IDF analysis of any number of documents"""

    def __init__(self, graphs: Dict[str, Graph]) -> None:
        self.graphs = 0
        self.nodes = 0

        self.graph_frequencies: Dict[str, int] = defaultdict(int)
        self.node_frequencies: Dict[str, int] = defaultdict(int)

        self.index: Dict[str, float] = defaultdict(lambda: 1)

        self._build_index(graphs)

    def _add_graph(self, graph: Graph) -> None:
        """Add a new document to the index"""

        self.graphs += 1

        for token in graph.tokens:
            self.graph_frequencies[token] += 1

        for node in graph.i_nodes.values():
            self.nodes += 1
            for token in node.tokens:
                self.node_frequencies[token] += 1

    def _get_graph_idf(self, token: str) -> float:
        """Calculate the idf value based on all saved documents"""

        return math.log10(self.graphs / self.graph_frequencies[token])

    def _get_node_idf(self, token: str) -> float:
        """Calculate the idf value based on all saved documents"""

        return math.log10(self.nodes / self.node_frequencies[token])

    def _build_index(self, graphs: Dict[str, Graph]) -> None:
        """Calculate and store the idf value for all tokens"""

        for graph in graphs.values():
            self._add_graph(graph)

        if config["token_weighting"] == "idf-graph":
            for item in self.graph_frequencies.keys():
                self.index[item] = self._get_graph_idf(item)
        elif config["token_weighting"] == "idf-node":
            for item in self.node_frequencies.keys():
                self.index[item] = self._get_node_idf(item)
        elif config["token_weighting"] == "sif":
            sif = SifWeights.get_instance()
            self.index = sif.weights
