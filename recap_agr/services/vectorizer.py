from __future__ import absolute_import, annotations

import gzip
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import scipy

from ..models.graph import Graph
from ..models.nlp import Embedding
from ..services import utils
from ..services.token_weighter import TokenWeighter

logger = logging.getLogger("recap")
config = utils.Config.get_instance()


@dataclass
class Vectorizer(object):
    embeddings: Dict[str, Embedding]
    token_weighter: TokenWeighter
    graphs: Dict[str, Graph]
    query_graphs: Dict[str, Graph]

    def __post_init__(self) -> None:
        for emb in self.embeddings.values():
            emb.preprocess(self.graphs, self.query_graphs)

    def get_graph_vectors(self, graph: Graph) -> None:
        """Save all node vectors as well as the graph vector"""

        for node in graph.i_nodes.values():
            node.vector, node.vectors = self._get_concatenated_embeddings(node.tokens)

        graph.vector, graph.vectors = self._get_concatenated_embeddings(graph.tokens)

    def _get_concatenated_embeddings(self, tokens: List[str]) -> np.ndarray:
        """Compute a vector based on a set of tokens

        For each loaded word embedding, a vector will be generated.
        All vectors will then be concatenated
        """

        aggregated_embeddings: List[np.ndarray] = []
        individual_embeddings: List[List[np.ndarray]] = []

        for word_embedding in self.embeddings.values():
            aggregated, individual = word_embedding.query(tokens, self.token_weighter)
            aggregated_embeddings.append(aggregated)
            individual_embeddings.append(individual)

        # It is necessary to transpose the results
        individual_embeddings = [list(emb) for emb in zip(*individual_embeddings)]

        aggregated_concat = np.concatenate(aggregated_embeddings, axis=0)
        individual_concat = [
            np.concatenate(emb, axis=0) for emb in individual_embeddings
        ]

        return aggregated_concat, individual_concat
