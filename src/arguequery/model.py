from dataclasses import dataclass

import arguebuf
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph
from cbrkit.model.graph import (
    Graph,
    SerializedEdge,
    SerializedGraph,
)


@dataclass(frozen=True, slots=True)
class SchemeData:
    scheme: arguebuf.Scheme | None


type AtomData = str
type NodeData = AtomData | SchemeData
type EdgeData = None
type GraphData = str
type KeyType = str


def load_graph(obj: AnnotatedGraph) -> Graph[KeyType, NodeData, EdgeData, GraphData]:
    g = arguebuf.load.protobuf(obj.graph)

    atom_nodes: dict[KeyType, NodeData] = {
        key: value.plain_text for key, value in g.atom_nodes.items()
    }
    scheme_nodes: dict[KeyType, NodeData] = {
        key: SchemeData(value.scheme) for key, value in g.scheme_nodes.items()
    }
    edges: dict[KeyType, SerializedEdge[KeyType, EdgeData]] = {
        key: SerializedEdge(source=value.source.id, target=value.target.id, value=None)
        for key, value in g.edges.items()
    }

    return Graph.load(
        SerializedGraph(
            nodes={**atom_nodes, **scheme_nodes},
            edges=edges,
            value=obj.text,
        )
    )
