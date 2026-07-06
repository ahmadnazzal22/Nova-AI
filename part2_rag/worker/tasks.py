import os
import tempfile
import time

from .celery_app import celery_app
from ..logger import get_logger

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def ingest_document(self, file_key: str, filename: str, chunk_size: int = 512, chunk_overlap: int = 64, namespace: str = ""):
    from ..storage.object_store import get_object_store
    from ..chunker import ingest_document as chunk_doc

    logger.info("Worker starting ingest: %s (key=%s)", filename, file_key)
    store = get_object_store()
    data = store.download(file_key)
    if not data:
        raise ValueError(f"File not found: {file_key}")

    ext = os.path.splitext(filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(data)
        tmp.close()
        chunks = chunk_doc(tmp.name, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            return {"filename": filename, "chunks": 0, "total_docs": 0}

        from ..retrieval.qdrant_store import QdrantStore
        from ..custom_embeddings import TransformerEmbeddings

        embeddings = TransformerEmbeddings()
        qdrant = QdrantStore(embedding_fn=embeddings.embed_query)
        qdrant.add_texts(chunks, namespace=namespace)
        total = qdrant.count()

        logger.info("Worker ingest complete: %s -> %d chunks (total: %d)", filename, len(chunks), total)
        return {"filename": filename, "chunks": len(chunks), "total_docs": total}
    finally:
        os.unlink(tmp.name)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def generate_embeddings_batch(self, texts: list[str], namespace: str = ""):
    from ..custom_embeddings import TransformerEmbeddings
    from ..retrieval.qdrant_store import QdrantStore

    embeddings = TransformerEmbeddings()
    qdrant = QdrantStore(embedding_fn=embeddings.embed_query)
    ids = qdrant.add_texts(texts, namespace=namespace)
    logger.info("Batch embedded %d texts -> %d ids", len(texts), len(ids))
    return {"embedded": len(ids)}


@celery_app.task
def clean_expired_sessions():
    from ..database import get_session
    logger.info("Cleaning expired sessions...")
    return {"cleaned": 0}
