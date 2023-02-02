import typing as t

import arguebuf as ag
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph


class Graph(ag.Graph):
    text: str


def from_protobuf(obj: AnnotatedGraph) -> Graph:
    g = t.cast(
        Graph, ag.from_protobuf(obj.graph, config=ag.ConverterConfig(GraphClass=Graph))
    )
    g.text = obj.text

    return g
