from __future__ import absolute_import, annotations

from dataclasses import dataclass

import arguebuf as ag
import numpy as np


@dataclass
class Result(object):
    """Store a graph and its similarity to the input query"""

    graph: ag.Graph
    similarity: float
