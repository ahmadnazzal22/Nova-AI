import os
import tempfile
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from ...schemas.schemas import IngestResponse
from ...chunker import ingest_document, SUPPORTED_EXTS
from ...exceptions import RAGError
from ...logger import get_logger
from ...orchestrator.rag_orchestrator import get_orchestrator

logger = get_logger(__name__)
router = APIRouter(tags=["RAG"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    file: UploadFile = File(...),
    chunk_size: int = Form(512, ge=64, le=4096),
    chunk_overlap: int = Form(64, ge=0, le=512),
):
    fid = uuid.uuid4().hex[:8]
    fn = file.filename or "unknown"
    ext = os.path.splitext(fn)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Supported: {', '.join(SUPPORTED_EXTS)}")
    tmp = None
    try:
        data = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(data)
        tmp.close()
        logger.info("[%s] Ingest: %s (%d B)", fid, fn, len(data))
        chunks = ingest_document(tmp.name, chunk_size=chunk_size, overlap=chunk_overlap)

        if chunks:
            from ...custom_embeddings import TransformerEmbeddings
            from ...retrieval.qdrant_store import QdrantStore
            embeddings = TransformerEmbeddings()
            qdrant = QdrantStore(embedding_fn=embeddings.embed_query)
            qdrant.add_texts(chunks)
            total = qdrant.count()
        else:
            total = 0

        return IngestResponse(filename=fn, chunks=len(chunks) if chunks else 0, total_docs=total)
    except RAGError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
