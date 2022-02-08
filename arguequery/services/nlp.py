from __future__ import annotations

import itertools
import typing as t

import arguebuf as ag
import grpc
import nlp_service.similarity
import numpy as np
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc
from arguequery.config import config
from arguequery.models.ontology import Ontology

_vector_cache = {}

_model_params = {
    "spacy": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        spacy_model="en_core_web_lg",
    ),
    "spacy_trf": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        spacy_model="en_core_web_trf",
    ),
    "use": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_USE,
                # model_name="https://tfhub.dev/google/universal-sentence-encoder-large/5",
                model_name="https://tfhub.dev/google/universal-sentence-encoder/4",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
    "sbert": nlp_pb2.NlpConfig(
        language=config.nlp.lang,
        embedding_models=[
            nlp_pb2.EmbeddingModel(
                model_type=nlp_pb2.EMBEDDING_TYPE_SBERT,
                model_name="stsb-mpnet-base-v2",
                pooling=nlp_pb2.POOLING_MEAN,
            )
        ],
    ),
}


_channel = grpc.insecure_channel(
    config.nlp.url, [("grpc.lb_policy_name", "round_robin")]
)
_client = nlp_pb2_grpc.NlpServiceStub(_channel)


def _vectors(texts: t.Iterable[str]) -> t.Tuple[np.ndarray, ...]:
    if new_texts := [text for text in texts if text not in _vector_cache]:
        levels = (
            [nlp_pb2.EMBEDDING_LEVEL_TOKENS]
            if config.nlp.token_vectors
            else [nlp_pb2.EMBEDDING_LEVEL_DOCUMENT]
        )

        res = _client.Vectors(
            nlp_pb2.VectorsRequest(
                texts=new_texts,
                embedding_levels=levels,
                config=_model_params[config.nlp.embedding],
            )
        )

        new_vectors = (
            tuple(
                tuple(np.array(token.vector) for token in x.tokens) for x in res.vectors
            )
            if config.nlp.token_vectors
            else tuple(np.array(x.document.vector) for x in res.vectors)
        )

        _vector_cache.update(dict(zip(new_texts, new_vectors)))

    return tuple(_vector_cache[text] for text in texts)


def _vector(text: str) -> np.ndarray:
    return _vectors([text])[0]


def _similarities(text_tuples: t.Iterable[t.Tuple[str, str]]) -> t.Tuple[float, ...]:
    # We first transform all text tuples into a long list to make the server processing faster
    vecs = _vectors(itertools.chain.from_iterable(text_tuples))

    # Then, we construct an iterator over all received vectors
    vecs_iter = iter(vecs)

    # By using the same iterator twice in the zip function, we always iterate over two entries per loop
    return tuple(
        getattr(nlp_service.similarity, config.nlp.similarity)(vec1, vec2)
        for vec1, vec2 in zip(vecs_iter, vecs_iter)
    )


def _similarity(text1: str, text2: str) -> float:
    return _similarities([(text1, text2)])[0]


GraphElement = t.Union[ag.Node, ag.Edge, ag.Graph]


def _graph2text(g: ag.Graph) -> str:
    return " ".join(node.plain_text for node in g.atom_nodes.values())


def similarities(
    objs: t.Iterable[t.Tuple[GraphElement, GraphElement]]
) -> t.Tuple[float, ...]:
    result = []

    for obj1, obj2 in objs:
        if isinstance(obj1, ag.AtomNode) and isinstance(obj2, ag.AtomNode):
            result.append(_similarity(obj1.plain_text, obj2.plain_text))

        elif isinstance(obj1, ag.SchemeNode) and isinstance(obj2, ag.SchemeNode):
            if obj1.type and obj2.type and config.nlp.schemes:
                if (
                    obj1.type == ag.SchemeType.SUPPORT
                    and obj2.type == ag.SchemeType.SUPPORT
                    and config.nlp.ontology
                ):
                    ontology = Ontology.instance()
                    result.append(
                        ontology.similarity(
                            obj1.argumentation_scheme, obj2.argumentation_scheme
                        )
                    )

                elif obj1.type == obj2.type:
                    result.append(1.0)

                else:
                    result.append(0.0)

            else:
                result.append(1.0)

        elif isinstance(obj1, ag.Edge) and isinstance(obj2, ag.Edge):
            if (source_sim := similarity(obj1.source, obj2.source)) and (
                target_sim := similarity(obj1.target, obj2.target)
            ):
                result.append(0.5 * (source_sim + target_sim))

        elif isinstance(obj1, ag.Graph) and isinstance(obj2, ag.Graph):
            result.append(_similarity(_graph2text(obj1), _graph2text(obj2)))

        else:
            result.append(0.0)

    return tuple(result)


def similarity(obj1: GraphElement, obj2: GraphElement) -> float:
    return similarities([(obj1, obj2)])[0]