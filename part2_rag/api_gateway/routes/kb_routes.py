import json
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel

from ...schemas.schemas import ChatRequest
from ...auth.middleware import get_current_user, require_admin
from ...database import session_scope, KnowledgeBaseRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Knowledge Base"])


class KBCreateRequest(BaseModel):
    name: str
    description: str = ""
    is_public: bool = False


class KBUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None


class KBPermissionRequest(BaseModel):
    user_id: int
    permission: str = "read"


def _get_kb_repo():
    from ...database import session_scope, KnowledgeBaseRepository
    return KnowledgeBaseRepository


@router.get("/kb")
def list_kbs(current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kbs = repo.list_for_user(uid)
        return [
            {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "user_id": kb.user_id,
                "collection_name": kb.collection_name,
                "is_public": kb.is_public,
                "is_owner": kb.user_id == uid,
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
            }
            for kb in kbs
        ]


@router.post("/kb")
def create_kb(req: KBCreateRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.create(
            user_id=uid,
            name=req.name,
            description=req.description,
            is_public=req.is_public,
        )
        return {"id": kb.id, "name": kb.name, "collection_name": kb.collection_name}


@router.get("/kb/{kb_id}")
def get_kb(kb_id: int, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        if not repo.check_permission(kb_id, uid, "read"):
            raise HTTPException(403, "Permission denied")
        return {
            "id": kb.id,
            "name": kb.name,
            "description": kb.description,
            "user_id": kb.user_id,
            "collection_name": kb.collection_name,
            "is_public": kb.is_public,
            "count": _count_docs(kb.collection_name),
            "is_owner": kb.user_id == uid,
            "created_at": kb.created_at.isoformat(),
            "updated_at": kb.updated_at.isoformat(),
        }


@router.put("/kb/{kb_id}")
def update_kb(kb_id: int, req: KBUpdateRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.update(kb_id, uid, **req.model_dump(exclude_none=True))
        if not kb:
            raise HTTPException(404, "Knowledge base not found or not owner")
        return {"id": kb.id, "name": kb.name, "description": kb.description, "is_public": kb.is_public}


@router.delete("/kb/{kb_id}")
def delete_kb(kb_id: int, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        col_name = kb.collection_name
        ok = repo.delete(kb_id, uid)
        if not ok:
            raise HTTPException(403, "Only the owner can delete")
    # Delete vector store documents for this KB
    try:
        from ...retrieval.qdrant_store import QdrantStore
        vs = QdrantStore()
        vs.delete_namespace(col_name)
    except Exception as e:
        logger.warning("Vector cleanup for %s failed: %s", col_name, e)
    return {"deleted": True}


@router.get("/kb/{kb_id}/permissions")
def list_permissions(kb_id: int, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        if kb.user_id != uid:
            raise HTTPException(403, "Only owner can manage permissions")
        perms = repo.list_permissions(kb_id)
        return [
            {"id": p.id, "user_id": p.user_id, "permission": p.permission, "created_at": p.created_at.isoformat()}
            for p in perms
        ]


@router.post("/kb/{kb_id}/permissions")
def add_permission(kb_id: int, req: KBPermissionRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        if kb.user_id != uid:
            raise HTTPException(403, "Only owner can manage permissions")
        perm = repo.add_permission(kb_id, req.user_id, req.permission)
        if not perm:
            raise HTTPException(400, "Failed to add permission")
        return {"id": perm.id, "user_id": perm.user_id, "permission": perm.permission}


@router.delete("/kb/{kb_id}/permissions/{perm_id}")
def remove_permission(kb_id: int, perm_id: int, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        if kb.user_id != uid:
            raise HTTPException(403, "Only owner can manage permissions")
        ok = repo.remove_permission(perm_id)
        if not ok:
            raise HTTPException(404, "Permission not found")
        return {"deleted": True}


@router.post("/kb/{kb_id}/ingest")
async def ingest_file(
    kb_id: int,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    uid = current_user.get("id") if current_user else None
    if not uid:
        raise HTTPException(401, "Authentication required")
    with session_scope() as session:
        repo = KnowledgeBaseRepository(session)
        if not repo.check_permission(kb_id, uid, "write"):
            raise HTTPException(403, "Permission denied")
        kb = repo.get(kb_id)
        if not kb:
            raise HTTPException(404, "Knowledge base not found")
        col_name = kb.collection_name

    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    chunks = _chunk_text(text)

    from ...retrieval.qdrant_store import QdrantStore
    from ...custom_embeddings import TransformerEmbeddings
    vs = QdrantStore(embedding_fn=TransformerEmbeddings().embed_query)
    metas = [{"title": file.filename or "upload", "source": "kb_upload"} for _ in chunks]
    ids = vs.add_texts(chunks, metadata=metas, namespace=col_name)
    return {"chunks": len(chunks), "ids": ids[:5], "total_ids": len(ids)}


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start + chunk_size // 2:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


def _count_docs(collection_name: str) -> int:
    try:
        from ...retrieval.qdrant_store import QdrantStore
        vs = QdrantStore()
        return vs.count()
    except Exception:
        return 0
