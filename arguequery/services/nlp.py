from __future__ import annotations

import itertools
import typing as t
from inspect import isgenerator

import arguebuf as ag
import grpc
import nlp_service
import numpy as np
from arg_services.cbr.v1beta.retrieval_pb2 import SchemeHandling
from arg_services.nlp.v1 import nlp_pb2, nlp_pb2_grpc

from arguequery.models.ontology import Ontology

GraphElement = t.Union[ag.AbstractNode, ag.Edge, str]
Vector = nlp_service.types.NumpyVector


class Nlp:
    _client: nlp_pb2_grpc.NlpServiceStub
    _config: nlp_pb2.NlpConfig
    scheme_handling: SchemeHandling.ValueType
    _vector_cache: dict[str, Vector] = {}

    def __init__(
        self,
        address: str,
        config: nlp_pb2.NlpConfig,
        scheme_handling: SchemeHandling.ValueType,
    ):
        channel = grpc.insecure_channel(
            address, [("grpc.lb_policy_name", "round_robin")]
        )
        self._client = nlp_pb2_grpc.NlpServiceStub(channel)
        self._config = config
        self.scheme_handling = scheme_handling

    @property
    def use_token_vectors(self):
        return self._config.similarity_method in (
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_DICE,
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_JACCARD,
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_MAXPOOL_JACCARD,
            nlp_pb2.SimilarityMethod.SIMILARITY_METHOD_DYNAMAX_OTSUKA,
        )

    def _vectors(self, texts: t.Iterable[str]) -> t.Tuple[Vector, ...]:
        if isgenerator(texts):
            texts = list(texts)

        if new_texts := [text for text in texts if text not in self._vector_cache]:
            levels = (
                [nlp_pb2.EMBEDDING_LEVEL_TOKENS]
                if self.use_token_vectors
                else [nlp_pb2.EMBEDDING_LEVEL_DOCUMENT]
            )

            res: nlp_pb2.VectorsResponse = self._client.Vectors(
                nlp_pb2.VectorsRequest(
                    texts=new_texts,
                    embedding_levels=levels,
                    config=self._config,
                )
            )

            new_vectors: tuple[Vector, ...] = (
                tuple(
                    tuple(np.array(token.vector) for token in x.tokens)
                    for x in res.vectors
                )
                if self.use_token_vectors
                else tuple(np.array(x.document.vector) for x in res.vectors)
            )

            self._vector_cache.update(zip(new_texts, new_vectors))

        return tuple(self._vector_cache[text] for text in texts)

    def _similarities(
        self, text_tuples: t.Iterable[t.Tuple[str, str]]
    ) -> t.Tuple[float, ...]:
        # We first transform all text tuples into a long list to make the server processing faster
        texts_chain = list(itertools.chain.from_iterable(text_tuples))
        vecs = self._vectors(texts_chain)

        assert len(vecs) == len(texts_chain)

        # Then, we construct an iterator over all received vectors
        vecs_iter = iter(vecs)

        # By using the same iterator twice in the zip function, we always iterate over two entries per loop
        return tuple(
            nlp_service.similarity.proto_mapping[self._config.similarity_method](
                vec1, vec2
            )
            for vec1, vec2 in zip(vecs_iter, vecs_iter)
        )

    def _similarity(self, text1: str, text2: str) -> float:
        return self._similarities([(text1, text2)])[0]

    def similarities(
        self, objs: t.Iterable[t.Tuple[GraphElement, GraphElement]]
    ) -> t.Tuple[float, ...]:
        result = []

        for obj1, obj2 in objs:
            if isinstance(obj1, ag.AtomNode) and isinstance(obj2, ag.AtomNode):
                result.append(self._similarity(obj1.plain_text, obj2.plain_text))

            elif isinstance(obj1, ag.SchemeNode) and isinstance(obj2, ag.SchemeNode):
                if self.scheme_handling == SchemeHandling.SCHEME_HANDLING_BINARY:
                    if type(obj1.scheme) == type(obj2.scheme):
                        result.append(1.0)
                    else:
                        result.append(0.0)

                elif self.scheme_handling == SchemeHandling.SCHEME_HANDLING_TAXONOMY:
                    if isinstance(obj1.scheme, ag.Support) and isinstance(
                        obj2.scheme, ag.Support
                    ):
                        ontology = Ontology.instance()
                        result.append(ontology.similarity(obj1.scheme, obj2.scheme))
                    elif type(obj1.scheme) == type(obj2.scheme):
                        result.append(1.0)
                    else:
                        result.append(0.0)

                else:
                    result.append(1.0)

            elif isinstance(obj1, ag.Edge) and isinstance(obj2, ag.Edge):
                result.append(
                    0.5
                    * (
                        self.similarity(obj1.source, obj2.source)
                        + self.similarity(obj1.target, obj2.target)
                    )
                )

            else:
                result.append(0.0)

        return tuple(result)

    def similarity(self, obj1: GraphElement, obj2: GraphElement) -> float:
        return self.similarities([(obj1, obj2)])[0]
