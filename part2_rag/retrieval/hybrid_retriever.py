"""
Hybrid Retrieval: Dense vector search + BM25 keyword search with weighted fusion.
"""
import re
from typing import Any, Optional

from .bm25 import BM25Okapi
from ..logger import get_logger

logger = get_logger(__name__)

_DENSE_WEIGHT = 0.7
_SPARSE_WEIGHT = 0.3


class HybridRetriever:
    def __init__(
        self,
        vector_store: Any = None,
        embedding_fn: Any = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._vector_store = vector_store
        self._embedding_fn = embedding_fn
        self._bm25 = BM25Okapi(k1=k1, b=b)
        self._documents: list[dict] = []
        self._max_docs = 10000

    @property
    def bm25(self) -> BM25Okapi:
        return self._bm25

    def rebuild_index(self, documents: list[dict]):
        self._documents = documents[:self._max_docs]
        texts = [d.get("text", d.get("content", d.get("snippet", ""))) for d in self._documents]
        self._bm25.index(texts)
        logger.info("BM25 index rebuilt: %d documents", len(self._documents))

    def add_documents(self, new_docs: list[dict]):
        self._documents.extend(new_docs)
        if len(self._documents) > self._max_docs:
            self._documents = self._documents[-self._max_docs:]
        texts = [d.get("text", d.get("content", d.get("snippet", ""))) for d in self._documents]
        self._bm25.index(texts)

    def search(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = _DENSE_WEIGHT,
        sparse_weight: float = _SPARSE_WEIGHT,
        extra_queries: Optional[list[str]] = None,
    ) -> list[dict]:
        all_queries = [query] + (extra_queries or [])

        dense_results: list[tuple[dict, float]] = []
        if self._vector_store is not None:
            dense_results = self._vector_search(all_queries, k)

        sparse_scores: dict[int, float] = {}
        if self._bm25.built:
            sparse_scores = self._sparse_search(all_queries)

        merged = self._fuse_results(
            dense_results, sparse_scores, dense_weight, sparse_weight, k
        )
        return merged[:k]

    def _vector_search(
        self, queries: list[str], k: int
    ) -> list[tuple[dict, float]]:
        results: list[tuple[dict, float]] = []
        seen_texts: set[str] = set()

        for q in queries:
            try:
                if hasattr(self._vector_store, "similarity_search_with_score"):
                    docs_scores = self._vector_store.similarity_search_with_score(q, k=k)
                elif hasattr(self._vector_store, "similarity_search"):
                    docs = self._vector_store.similarity_search(q, k=k)
                    docs_scores = [(d, 0.5) for d in docs]
                else:
                    docs = self._vector_store.invoke(q)
                    docs_scores = [(d, 0.5) for d in docs] if isinstance(docs, list) else []

                for doc, score in docs_scores:
                    text = ""
                    if hasattr(doc, "page_content"):
                        text = doc.page_content
                    elif isinstance(doc, dict):
                        text = doc.get("text", doc.get("content", doc.get("snippet", "")))
                    key = text[:200].strip().lower() if text else ""
                    if key and key not in seen_texts:
                        seen_texts.add(key)
                        if not isinstance(doc, dict):
                            doc = {"text": text, "_source": doc}
                        results.append((doc, float(score)))
            except Exception as e:
                logger.debug("Vector search failed for query '%s': %s", q[:40], e)

        results.sort(key=lambda x: -x[1])
        return results[:k * 2]

    def _sparse_search(self, queries: list[str]) -> dict[int, float]:
        scores: dict[int, float] = {}
        weight = 1.0 / len(queries)
        for q in queries:
            q_scores = self._bm25.get_scores(q)
            for i, s in enumerate(q_scores):
                scores[i] = scores.get(i, 0.0) + s * weight
        return scores

    def _fuse_results(
        self,
        dense: list[tuple[dict, float]],
        sparse: dict[int, float],
        dense_weight: float,
        sparse_weight: float,
        k: int,
    ) -> list[dict]:
        fused: dict[str, tuple[dict, float, float, int]] = {}

        for doc, score in dense:
            text = doc.get("text", doc.get("content", doc.get("snippet", "")))
            key = text[:200].strip().lower()
            if key not in fused:
                fused[key] = (doc, score, 0.0, 0)
            else:
                existing = fused[key]
                fused[key] = (doc, max(existing[1], score), existing[2], existing[3] + 1)

        if sparse:
            for idx, score in sparse.items():
                if idx < len(self._documents):
                    doc = self._documents[idx]
                    text = doc.get("text", doc.get("content", doc.get("snippet", "")))
                    key = text[:200].strip().lower()
                    if key not in fused:
                        fused[key] = (doc, 0.0, score, 0)
                    else:
                        existing = fused[key]
                        fused[key] = (existing[0], existing[1], max(existing[2], score), existing[3])

        results = []
        for key, (doc, dense_score, sparse_score, count) in fused.items():
            if dense_weight > 0 and sparse_weight > 0:
                d = dense_score
                s = sparse_score / max(1.0, max(
                    (v[2] for v in fused.values()), default=1.0
                ))
                combined = dense_weight * d + sparse_weight * s
            elif dense_weight > 0:
                combined = dense_score
            else:
                combined = sparse_score

            doc["_relevance_score"] = round(combined, 4)
            doc["_dense_score"] = round(dense_score, 4)
            doc["_sparse_score"] = round(sparse_score, 4)
            doc["_score_components"] = {
                "dense": round(dense_score, 4),
                "sparse": round(sparse_score, 4),
                "combined": round(combined, 4),
            }
            results.append((combined, doc))

        results.sort(key=lambda x: -x[0])
        return [doc for _, doc in results]
