import typing as t

import arguebuf as ag
from arguequery.config import config
from arguequery.services import nlp
from networkx.algorithms import isomorphism as morph


def _atom_label(node: ag.AtomNode) -> str:
    return "AtomNode"


def _scheme_label(node: ag.SchemeNode) -> str:
    label = "SchemeNode"

    if nlp.enforce_scheme_types and node.type:
        label += f": {node.type.value}"

        if nlp.use_scheme_ontology and node.argumentation_scheme:
            label += f": {node.argumentation_scheme.value}"

    return label


def run(cases: t.Mapping[str, ag.Graph], query: ag.Graph) -> t.Dict[str, float]:
    q = query.to_nx(atom_label=_atom_label, scheme_label=_scheme_label)
    similarities: t.Dict[str, float] = {}

    for case_id, case in cases.items():
        c = case.to_nx(atom_label=_atom_label, scheme_label=_scheme_label)

        # Search for subgraphs of 'c' in 'q'
        matcher = morph.DiGraphMatcher(c, q, node_match=lambda x, y: x.label == y.label)
        mappings: t.List[t.Mapping[str, str]] = list(
            matcher.subgraph_monomorphisms_iter()
        )
        mapping_similarities: t.Dict[int, float] = {}

        for i, mapping in enumerate(mappings):
            sim = 0

            for case_node_id, query_node_id in mapping.items():
                case_node = case.nodes[case_node_id]
                query_node = query.nodes[query_node_id]

                if isinstance(case_node, ag.AtomNode) and isinstance(
                    query_node, ag.AtomNode
                ):
                    sim += nlp.similarity(case_node, query_node)

            mapping_similarities[i] = sim / len(query.atom_nodes)

        _, best_sim = max(mapping_similarities.items(), key=lambda x: x[1])
        similarities[case_id] = best_sim

    return similarities


# TODO: Return mapping via gRPC
# TODO: Add this mapping method to gRPC
