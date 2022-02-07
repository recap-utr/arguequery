from __future__ import absolute_import, annotations

import gzip
import json
import logging
import os
import subprocess
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import gensim
import nltk
import numpy as np

from ..models.graph import Edge, Graph, Node
from ..models.nlp import Embedding
from ..services import utils

logger = logging.getLogger("recap")
from arguequery.config import config


def jsonobj2graph(json_data: Dict[str, Any], filename: str) -> Graph:
    """Convert a given dict to a graph structure"""

    i_nodes_dict = {}
    s_nodes_dict = {}
    edges_dict = {}

    json_nodes = json_data["nodes"]
    json_edges = json_data["edges"]
    last_node_id = 1

    # Add all nodes to dictionary with their id's as keys
    for json_node in json_nodes:
        node = Node(
            int(json_node.get("nodeID") or json_node.get("id")),
            json_node["text"],
            json_node["type"],
        )

        if node.type_ == "I":
            i_nodes_dict[node.id_] = node
            node.tokens = utils.get_tokens(node.text)
        else:
            s_nodes_dict[node.id_] = node

        last_node_id = node.id_

    nodes_dict = {**i_nodes_dict, **s_nodes_dict}

    id_generator = utils.generate_id(last_node_id)

    # Add all edges to dictionary with their id's as keys
    for json_edge in json_edges:
        edge_id = int(
            json_edge.get("edgeID") or json_edge.get("id") or next(id_generator)
        )
        from_id = int(json_edge.get("fromID") or json_edge.get("from").get("id"))
        to_id = int(json_edge.get("toID") or json_edge.get("to").get("id"))

        edge = Edge(edge_id, nodes_dict[from_id], nodes_dict[to_id])

        edges_dict[edge.id_] = edge

    # Create graph and save filename along with it
    graph = Graph(nodes_dict, i_nodes_dict, s_nodes_dict, edges_dict, filename)
    graph.tokens = utils.get_tokens(graph.text)

    return graph


def jsonfile2graph(
    filepath: str,
) -> Graph:
    """Read a single json file and import the contents to a graph model"""

    graph = None
    basename = os.path.basename(filepath)

    with open(filepath, "r") as file:
        graph = jsonobj2graph(json.load(file), basename)

    return graph


def jsonfiles2graphs() -> Dict[str, Graph]:
    """Iterate over JSON files in directory and create graphs"""

    graphs_dict: Dict[str, Graph] = {}

    # Iterate over directory and create graph for each file
    for filename in os.listdir(config["casebase_folder"]):
        if not filename.endswith(".json"):
            continue

        fullname: str = os.path.join(config["casebase_folder"], filename)
        graph = jsonfile2graph(fullname)
        graphs_dict[graph.filename] = graph

    return graphs_dict
