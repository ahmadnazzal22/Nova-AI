import json
import time
import uuid
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from ...schemas.schemas import QueryRequest, QueryResponse, ChatRequest, LiveQueryResponse, HealthResponse, IngestResponse
from ...auth.middleware import get_current_user, require_user
from ...exceptions import RAGError, LLMError
from ...logger import get_logger
from ...orchestrator.rag_orchestrator import get_orchestrator
from ...orchestrator.query_router import route_question
from ...monitoring.metrics import get_metrics
from ...cache.redis_cache import get_query_cache

logger = get_logger(__name__)
router = APIRouter(tags=["RAG"])


def _get_orch():
    orch = get_orchestrator()
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return orch


def _resolve_kb_namespace(kb_id: int | None) -> str:
    """Resolve kb_id to a Qdrant namespace (collection_name)."""
    if not kb_id:
        return ""
    try:
        from ...database import session_scope, KnowledgeBaseRepository
        with session_scope() as session:
            repo = KnowledgeBaseRepository(session)
            kb = repo.get(kb_id)
            if kb:
                return kb.collection_name
    except Exception:
        pass
    return ""


@router.get("/health", response_model=HealthResponse)
def health():
    metrics = get_metrics()
    return HealthResponse(status="ok", documents_indexed=0, model_loaded=True)


@router.get("/sources/{message_id}")
def get_message_sources(message_id: int, current_user: dict = Depends(get_current_user)):
    """Retrieve sources for a given message."""
    try:
        from ...database import session_scope, MessageRepository
        with session_scope() as session:
            mrepo = MessageRepository(session)
            msg = mrepo.get(message_id)
            if not msg:
                raise HTTPException(404, "Message not found")
            sources_raw = getattr(msg, "sources", None) or getattr(msg, "source_data", None) or "[]"
            if isinstance(sources_raw, str):
                sources = json.loads(sources_raw)
            else:
                sources = sources_raw
            return {"sources": sources, "message_id": message_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to get sources for msg %d: %s", message_id, e)
        return {"sources": [], "message_id": message_id}


@router.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    rid = uuid.uuid4().hex[:8]
    logger.info("[%s] Query: %.80s (kb_id=%d)", rid, req.question, kb_id)
    t0 = time.time()
    uid = current_user.get("id") if current_user else None
    try:
        ns = _resolve_kb_namespace(kb_id or None)
        result = _get_orch().kb_query(req.question, user_id=uid, kb_namespace=ns)
        elapsed = time.time() - t0
        get_metrics().record("query", elapsed)
        return QueryResponse(
            answer=result["answer"], sources=result.get("sources", []),
            citations=result.get("citations", []),
            related_questions=result.get("related_questions", []),
            conversation_id=conv_id, message_id=result.get("_msg_id", 0),
            cached=result.get("cached", False), tool_used=result.get("tool_used"),
        )
    except (RAGError, LLMError) as e:
        get_metrics().record("query", elapsed := time.time() - t0, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=QueryResponse)
def chat_endpoint(req: ChatRequest, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    return query_endpoint(QueryRequest(question=req.question), kb_id=kb_id, current_user=current_user)


@router.post("/live", response_model=LiveQueryResponse)
def live_endpoint(req: ChatRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    rid = uuid.uuid4().hex[:8]
    logger.info("[%s] Live: %.80s (kb_id=%d)", rid, req.question, kb_id)
    t0 = time.time()
    uid = current_user.get("id") if current_user else None
    try:
        result = _get_orch().live_query(req.question, user_id=uid)
        elapsed = time.time() - t0
        get_metrics().record("live_query", elapsed)
        return LiveQueryResponse(
            answer=result.get("answer", ""), sources=result.get("sources", []),
            citations=result.get("citations", []),
            live=result.get("live", True), cached=result.get("cached", False),
            related_questions=result.get("related_questions", []),
            conversation_id=conv_id, message_id=result.get("_msg_id", 0),
        )
    except (RAGError, LLMError) as e:
        get_metrics().record("live_query", time.time() - t0, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/live/stream")
def live_stream_endpoint(req: ChatRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    def _wrapped():
        answer = ""
        _mid = 0
        for event in _get_orch().live_stream(req.question, user_id=uid):
            yield f"data: {event}\n\n"
            try:
                p = json.loads(event)
                if p["type"] == "done":
                    answer = p.get("data", "")
            except Exception:
                pass
        if uid and answer:
            try:
                from ...database import session_scope, ConversationRepository, MessageRepository
                with session_scope() as session:
                    crepo = ConversationRepository(session)
                    mrepo = MessageRepository(session)
                    nc = conv_id
                    if not nc:
                        conv = crepo.create(uid, title=req.question[:80])
                        nc = conv.id
                    mrepo.add(nc, "user", req.question, model="live")
                    msg = mrepo.add(nc, "assistant", answer, live=True, model="live")
                    _mid = msg.id
            except Exception as e:
                logger.warning("Failed to save live stream: %s", e)
        if _mid:
            yield f"data: {json.dumps({'type': 'msg_id', 'data': _mid})}\n\n"
    return StreamingResponse(
        _wrapped(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    ns = _resolve_kb_namespace(kb_id or None)
    return StreamingResponse(
        _stream_chat(req.question, uid, conv_id, kb_namespace=ns),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/query/stream")
def query_stream_endpoint(req: QueryRequest, kb_id: int = 0):
    ns = _resolve_kb_namespace(kb_id or None)
    return StreamingResponse(
        _get_orch().stream_query(req.question, kb_namespace=ns),
        media_type="text/event-stream",
    )


def _stream_chat(question: str, uid: int | None, conv_id: int, kb_namespace: str = ""):
    t0 = time.time()
    tokens = []
    for chunk in _get_orch().stream_query(question, user_id=uid, kb_namespace=kb_namespace):
        if chunk.startswith("__CITATIONS__"):
            citations = chunk[len("__CITATIONS__"):]
            yield f"data: {json.dumps({'type': 'citations', 'data': json.loads(citations)})}\n\n"
            continue
        tokens.append(chunk)
        yield f"data: {json.dumps({'token': chunk})}\n\n"
    elapsed = time.time() - t0
    answer = "".join(tokens)
    yield f"data: {json.dumps({'done': True, 'conv_id': conv_id, 'msg_id': 0})}\n\n"


@router.post("/ask", response_model=QueryResponse)
def ask_endpoint(req: ChatRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    """Unified endpoint with auto-routing: live / research / rag."""
    rid = uuid.uuid4().hex[:8]
    uid = current_user.get("id") if current_user else None
    ns = _resolve_kb_namespace(kb_id or None)
    logger.info("[%s] Ask: %.80s (user=%s)", rid, req.question, uid)
    t0 = time.time()
    try:
        result = _get_orch().ask(req.question, user_id=uid, kb_namespace=ns)
        elapsed = time.time() - t0
        get_metrics().record("ask", elapsed)
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            citations=result.get("citations", []),
            related_questions=result.get("related_questions", []),
            conversation_id=conv_id,
            message_id=result.get("_msg_id", 0),
            cached=result.get("cached", False),
            tool_used=result.get("route", result.get("tool_used")),
        )
    except (RAGError, LLMError) as e:
        get_metrics().record("ask", time.time() - t0, error=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/stream")
def ask_stream_endpoint(req: ChatRequest, conv_id: int = 0, kb_id: int = 0, current_user: dict = Depends(get_current_user)):
    """Unified streaming endpoint with auto-routing."""
    uid = current_user.get("id") if current_user else None
    ns = _resolve_kb_namespace(kb_id or None)
    logger.info("Ask stream: %.80s (user=%s, kb_id=%d)", req.question, uid, kb_id)

    def _wrapped():
        answer = ""
        _mid = 0
        _nc = conv_id
        for event in _get_orch().ask_stream(req.question, user_id=uid, kb_namespace=ns):
            yield f"data: {event}\n\n"
            try:
                p = json.loads(event)
                if p.get("type") == "research_token":
                    answer += p.get("token", "")
                elif p.get("type") in ("done",):
                    answer = p.get("data", answer)
                elif p.get("type") == "research_done":
                    answer = p.get("report", answer)
            except Exception:
                pass
        if uid and answer:
            try:
                from ...database import session_scope, ConversationRepository, MessageRepository
                with session_scope() as session:
                    crepo = ConversationRepository(session)
                    mrepo = MessageRepository(session)
                    if not _nc:
                        conv = crepo.create(uid, title=req.question[:80])
                        _nc = conv.id
                    mrepo.add(_nc, "user", req.question, model="ask")
                    msg = mrepo.add(_nc, "assistant", answer, live=False, model="ask")
                    _mid = msg.id
            except Exception as e:
                logger.warning("Failed to save ask conversation: %s", e)
        if _mid:
            yield f"data: {json.dumps({'type': 'msg_id', 'data': _mid})}\n\n"
        if _nc and _nc != conv_id:
            yield f"data: {json.dumps({'type': 'conv_id', 'data': _nc})}\n\n"

    return StreamingResponse(
        _wrapped(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/route")
def route_endpoint(q: str = "", current_user: dict = Depends(get_current_user)):
    """Debug endpoint: returns the routing decision for a query."""
    from ..fast_path import is_fast_path
    if is_fast_path(q):
        return {"question": q, "route": "greeting"}
    return {"question": q, "route": route_question(q)}
