import os
import uuid
import time
from typing import Any

from ..logger import get_logger

logger = get_logger(__name__)


class QdrantStore:
    _shared_local_store: dict[str, list[dict]] = {}

    def __init__(self, url: str = "http://localhost:6333", collection: str = "rag_documents", vector_size: int = 384, embedding_fn: Any = None):
        self._client = None
        self._url = url
        self._collection = collection
        self._vector_size = vector_size
        self._embedding_fn = embedding_fn
        self._local_store = QdrantStore._shared_local_store
        self._enabled = False

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
                self._client = QdrantClient(url=self._url)
                self._models = models
                collections = self._client.get_collections().collections
                exists = any(c.name == self._collection for c in collections)
                if not exists:
                    self._client.create_collection(
                        collection_name=self._collection,
                        vectors_config=models.VectorParams(size=self._vector_size, distance=models.Distance.COSINE),
                    )
                    logger.info("Created Qdrant collection: %s (size=%d)", self._collection, self._vector_size)
                self._enabled = True
                logger.info("Qdrant connected: %s", self._url)
            except Exception as e:
                logger.warning("Qdrant unavailable, using local fallback: %s", e)
                self._enabled = False
        return self._client

    def _ensure_embedding(self, text: str) -> list[float]:
        if self._embedding_fn:
            return self._embedding_fn(text)
        return [0.0] * self._vector_size

    def add_texts(self, texts: list[str], metadata: list[dict] | None = None, namespace: str = "") -> list[str]:
        ids = [str(uuid.uuid4()) for _ in texts]
        vectors = [self._ensure_embedding(t) for t in texts]
        metas = metadata or [{} for _ in texts]
        for m in metas:
            m["_namespace"] = namespace
        if self._enabled:
            try:
                points = [
                    self._models.PointStruct(id=ids[i], vector=vectors[i], payload={"text": texts[i], **metas[i]})
                    for i in range(len(texts))
                ]
                self.client.upsert(collection_name=self._collection, points=points, wait=False)
                logger.info("Qdrant upsert: %d texts", len(texts))
            except Exception as e:
                logger.warning("Qdrant upsert failed: %s", e)
                self._enabled = False
        if not self._enabled:
            ns = namespace or "_default"
            if ns not in self._local_store:
                self._local_store[ns] = []
            for i, t in enumerate(texts):
                self._local_store[ns].append({"id": ids[i], "text": t, "vector": vectors[i], **metas[i]})
        return ids

    def similarity_search(self, query: str, k: int = 3, namespace: str = "") -> list[dict]:
        query_vec = self._ensure_embedding(query)
        ns_filter = namespace or "_default"
        if self._enabled:
            try:
                from qdrant_client.http import models as m
                filter_cond = m.Filter(must=[m.FieldCondition(key="_namespace", match=m.MatchValue(value=ns_filter))]) if namespace else None
                results = self.client.search(
                    collection_name=self._collection,
                    query_vector=query_vec,
                    limit=k,
                    query_filter=filter_cond,
                )
                return [{"text": r.payload.get("text", ""), "score": r.score, **{k: v for k, v in r.payload.items() if k != "text"}} for r in results]
            except Exception as e:
                logger.warning("Qdrant search failed: %s", e)
        store = self._local_store.get(ns_filter, [])
        scored = []
        for doc in store:
            vec = doc.get("vector", [])
            score = sum(a * b for a, b in zip(query_vec, vec)) if vec else 0
            scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [{**d, "score": s} for s, d in scored[:k]]

    def count(self) -> int:
        if self._enabled:
            try:
                return self.client.count(collection_name=self._collection).count
            except Exception:
                pass
        return sum(len(v) for v in self._local_store.values())

    def delete_namespace(self, namespace: str):
        if self._enabled:
            try:
                from qdrant_client.http import models as m
                self.client.delete(collection_name=self._collection, points_selector=m.FilterSelector(
                    filter=m.Filter(must=[m.FieldCondition(key="_namespace", match=m.MatchValue(value=namespace))])
                ))
            except Exception as e:
                logger.warning("Qdrant namespace delete failed: %s", e)
        self._local_store.pop(namespace, None)

    def health(self) -> dict:
        try:
            count = self.count()
            return {"status": "ok", "documents_indexed": count, "qdrant_enabled": self._enabled}
        except Exception as e:
            return {"status": "error", "error": str(e)}
