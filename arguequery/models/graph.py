import typing as t

import arguebuf as ag
from arg_services.cbr.v1beta.model_pb2 import AnnotatedGraph


class Graph(ag.Graph):
    text: str


def load(obj: AnnotatedGraph) -> Graph:
    g = t.cast(
        Graph, ag.load.protobuf(obj.graph, config=ag.load.Config(GraphClass=Graph))
    )
    g.text = obj.text

    return g
