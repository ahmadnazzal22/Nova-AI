from fastapi import APIRouter, HTTPException, Depends

from ...schemas.schemas import MemoryItem, MemoryStoreRequest
from ...auth.middleware import require_user
from ...database import get_session, session_scope, MemoryRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Memory"])


@router.get("/memories")
def list_memories(limit: int = 20, user: dict = Depends(require_user)):
    with get_session() as session:
        repo = MemoryRepository(session)
        memories = repo.get_recent(user["id"], limit=limit)
        return {"memories": [
            {"id": m.id, "key": m.key, "value": m.value, "importance": m.importance,
             "created_at": m.created_at.isoformat()}
            for m in memories
        ]}


@router.post("/memories")
def store_memory(req: MemoryStoreRequest, user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = MemoryRepository(session)
        mem = repo.store(user["id"], req.key, req.value, req.importance)
        return {"id": mem.id, "key": mem.key, "value": mem.value, "importance": mem.importance}


@router.delete("/memories/{key}")
def delete_memory(key: str, user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = MemoryRepository(session)
        if repo.delete_by_key(user["id"], key):
            return {"deleted": key}
        raise HTTPException(status_code=404, detail="Memory not found")


@router.get("/memories/search")
def search_memories(q: str, user: dict = Depends(require_user)):
    with get_session() as session:
        repo = MemoryRepository(session)
        results = repo.search(user["id"], q)
        return {"results": [
            {"id": m.id, "key": m.key, "value": m.value, "importance": m.importance,
             "created_at": m.created_at.isoformat()}
            for m in results
        ]}
