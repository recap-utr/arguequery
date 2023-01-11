from __future__ import annotations

import itertools
import typing as t
from inspect import isgenerator

import arguebuf as ag
import grpc
import nlp_service.similarity
import numpy as np
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc

from arguequery.config import config
from arguequery.models.ontology import Ontology

vector_cache = {}
nlp_config = None
use_scheme_ontology = None
enforce_scheme_types = None
query_text = None
case_texts = None

# _model_params = {
#     "spacy": nlp_pb2.NlpConfig(
#         language="en",
#         spacy_model="en_core_web_lg",
#     ),
#     "spacy_trf": nlp_pb2.NlpConfig(
#         language="en",
#         spacy_model="en_core_web_trf",
#     ),
#     "use": nlp_pb2.NlpConfig(
#         language="en",
#         embedding_models=[
#             nlp_pb2.EmbeddingModel(
#                 model_type=nlp_pb2.EMBEDDING_TYPE_USE,
#                 # model_name="https://tfhub.dev/google/universal-sentence-encoder-large/5",
#                 model_name="https://tfhub.dev/google/universal-sentence-encoder/4",
#                 pooling=nlp_pb2.POOLING_MEAN,
#             )
#         ],
#     ),
#     "sbert": nlp_pb2.NlpConfig(
#         language="en",
#         embedding_models=[
#             nlp_pb2.EmbeddingModel(
#                 model_type=nlp_pb2.EMBEDDING_TYPE_SBERT,
#                 model_name="stsb-mpnet-base-v2",
#                 pooling=nlp_pb2.POOLING_MEAN,
#             )
#         ],
#     ),
# }

address = config.nlp_address


def init_client():
    channel = grpc.insecure_channel(address, [("grpc.lb_policy_name", "round_robin")])
    return nlp_pb2_grpc.NlpServiceStub(channel)


client = init_client()

_use_token_vectors = lambda similarity_method: similarity_method in (
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_DICE,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_JACCARD,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_MAXPOOL_JACCARD,
    nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_OTSUKA,
)


def _vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    assert nlp_config is not None

    if isgenerator(texts):
        texts = list(texts)

    if new_texts := [text for text in texts if text not in vector_cache]:
        levels = (
            [nlp_pb2.EMBEDDING_LEVEL_TOKENS]
            if _use_token_vectors(nlp_config)
            else [nlp_pb2.EMBEDDING_LEVEL_DOCUMENT]
        )

        res = client.Vectors(
            nlp_pb2.VectorsRequest(
                texts=new_texts,
                embedding_levels=levels,
                config=nlp_config,
            )
        )

        new_vectors = (
            tuple(
                tuple(np.array(token.vector) for token in x.tokens) for x in res.vectors
            )
            if _use_token_vectors(nlp_config)
            else tuple(np.array(x.document.vector) for x in res.vectors)
        )

        vector_cache.update(dict(zip(new_texts, new_vectors)))

    return tuple(vector_cache[text] for text in texts)


def _vector(text: str) -> np.ndarray:
    return _vectors([text])[0]


def _similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[float, ...]:
    # We first transform all text tuples into a long list to make the server processing faster
    texts_chain = list(itertools.chain.from_iterable(text_tuples))
    vecs = _vectors(texts_chain)

    assert len(vecs) == len(texts_chain)
    assert nlp_config is not None

    # Then, we construct an iterator over all received vectors
    vecs_iter = iter(vecs)

    # By using the same iterator twice in the zip function, we always iterate over two entries per loop
    return tuple(
        nlp_service.similarity.proto_mapping[nlp_config.similarity_method](vec1, vec2)
        for vec1, vec2 in zip(vecs_iter, vecs_iter)
    )


def _similarity(text1: str, text2: str) -> float:
    return _similarities([(text1, text2)])[0]


GraphElement = t.Union[ag.Node, ag.Edge, ag.Graph, str]


def _graph2text(g: ag.Graph) -> str:
    # TODO
    return " ".join(node.plain_text for node in g.atom_nodes.values())


def similarities(
    objs: t.Iterable[t.Tuple[GraphElement, GraphElement]]
) -> t.Tuple[float, ...]:
    result = []

    for obj1, obj2 in objs:
        if isinstance(obj1, ag.AtomNode) and isinstance(obj2, ag.AtomNode):
            result.append(_similarity(obj1.plain_text, obj2.plain_text))

        elif isinstance(obj1, ag.SchemeNode) and isinstance(obj2, ag.SchemeNode):
            assert use_scheme_ontology is not None and enforce_scheme_types is not None

            if enforce_scheme_types:
                if type(obj1.scheme) == type(obj2.scheme):
                    if (
                        use_scheme_ontology
                        and isinstance(obj1.scheme, ag.Support)
                        and isinstance(obj2.scheme, ag.Support)
                    ):
                        ontology = Ontology.instance()
                        result.append(ontology.similarity(obj1.scheme, obj2.scheme))
                    else:
                        result.append(1.0)

                else:
                    result.append(0.0)

            else:
                result.append(1.0)

        elif isinstance(obj1, ag.Edge) and isinstance(obj2, ag.Edge):
            result.append(
                0.5
                * (
                    similarity(obj1.source, obj2.source)
                    + similarity(obj1.target, obj2.target)
                )
            )

        elif isinstance(obj1, (ag.Graph, str)) and isinstance(obj2, (ag.Graph, str)):
            if isinstance(obj1, ag.Graph):
                obj1 = _graph2text(obj1)

            if isinstance(obj2, ag.Graph):
                obj2 = _graph2text(obj2)

            result.append(_similarity(obj1, obj2))

        else:
            result.append(0.0)

    return tuple(result)


def similarity(obj1: GraphElement, obj2: GraphElement) -> float:
    return similarities([(obj1, obj2)])[0]
