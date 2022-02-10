import statistics
import typing as t

import arguebuf as ag
from arguequery.config import config
from arguequery.models.mapping import FacMapping, FacResults
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


def run(cases: t.Mapping[str, ag.Graph], query: ag.Graph) -> FacResults:
    q = query.to_nx(atom_label=_atom_label, scheme_label=_scheme_label)
    case_similarities: t.Dict[str, float] = {}
    case_mappings: t.Dict[str, t.Set[FacMapping]] = {}

    for case_id, case in cases.items():
        c = case.to_nx(atom_label=_atom_label, scheme_label=_scheme_label)

        # Search for subgraphs of 'c' in 'q'
        matcher = morph.DiGraphMatcher(
            c, q, node_match=lambda x, y: x["label"] == y["label"]
        )
        nx_mappings: t.List[t.Mapping[str, str]] = list(
            matcher.subgraph_monomorphisms_iter()
        )
        mappings: t.List[t.Set[FacMapping]] = []

        for nx_mapping in nx_mappings:
            mapping = set()

            for case_node_id, query_node_id in nx_mapping.items():
                case_node = case.nodes[case_node_id]
                query_node = query.nodes[query_node_id]

                if isinstance(case_node, ag.AtomNode) and isinstance(
                    query_node, ag.AtomNode
                ):
                    mapping.add(
                        FacMapping(
                            query_node_id,
                            case_node_id,
                            nlp.similarity(case_node, query_node),
                        )
                    )

            mappings.append(mapping)

        best_sim = 0
        best_mapping = set()

        if mappings:
            mapping_similarities = [
                statistics.mean(entry.similarity for entry in mapping)
                for mapping in mappings
            ]
            best_mapping_id, best_sim = max(
                enumerate(mapping_similarities),
                key=lambda x: x[1],
            )
            best_mapping = mappings[best_mapping_id]

        case_similarities[case_id] = best_sim
        case_mappings[case_id] = best_mapping

    return FacResults(case_similarities, case_mappings)
