import asyncio
import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError

from ...schemas.schemas import V1ChatRequest, V1ChatResponse, V1ErrorResponse, V1ErrorDetail, CitationItem
from ...auth.middleware import get_current_user
from ...exceptions import RAGError, LLMError
from ...logger import get_logger
from ...orchestrator.rag_orchestrator import get_orchestrator
from ...orchestrator.query_router import route_question
from ...monitoring.metrics import get_metrics

logger = get_logger(__name__)
router = APIRouter(tags=["V1"])

REQUEST_TIMEOUT = 120.0

ORCH = None

def _get_orch():
    global ORCH
    if ORCH is None:
        ORCH = get_orchestrator()
    if ORCH is None:
        raise HTTPException(status_code=503, detail=V1ErrorDetail(code="SERVICE_UNAVAILABLE", message="Orchestrator not initialized").model_dump())
    return ORCH


def _resolve_kb_namespace(kb_id: int) -> str:
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


def _extract_message(req: V1ChatRequest) -> str:
    raw = req.message
    if raw and raw.strip():
        return raw.strip()
    try:
        body = getattr(req, "_body", None) or req.model_dump()
    except Exception:
        body = {}
    for field in ("question", "text", "input", "msg", "query"):
        val = body.get(field) or getattr(req, field, None)
        if val and isinstance(val, str) and val.strip():
            logger.info("_extract_message: mapped from field '%s'", field)
            return val.strip()
    return ""


def _validate_message(message: str):
    if not message or not message.strip():
        err = V1ErrorDetail(code="EMPTY_MESSAGE", message="Message is required")
        raise HTTPException(status_code=400, detail=err.model_dump())


def _determine_mode(message: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        logger.info("v1 mode=explicit %s", requested_mode)
        return requested_mode
    route = route_question(message)
    logger.info("v1 mode=routed %s", route)
    return route


def _run_v1_chat(message: str, mode: str, user_id: int | None, kb_namespace: str) -> dict:
    stages = {}
    t_start = time.time()

    t0 = time.time()
    orch = _get_orch()
    stages["orchestrator"] = round((time.time() - t0) * 1000)

    t0 = time.time()
    answer_preview = ""
    if mode == "live":
        logger.info("v1 route=live: %.80s", message)
        result = orch.live_query(message, user_id=user_id)
        stages["live_query"] = round((time.time() - t0) * 1000)
        result["mode"] = "live"
        answer_preview = result.get("answer", "")[:100]
        logger.info("v1 live_result answer=%.100s sources=%d", answer_preview, len(result.get("sources", [])))
    elif mode == "research":
        logger.info("v1 route=research: %.80s", message)
        from ...research_agent import DeepResearchAgent
        agent = DeepResearchAgent()
        report = ""
        sources = []
        for event in agent.research(message, kb_namespace=kb_namespace):
            try:
                p = json.loads(event) if isinstance(event, str) else event
                if p.get("type") == "research_token":
                    report += p.get("token", "")
                elif p.get("type") == "research_done":
                    report = p.get("report", report)
                    sources = p.get("sources", [])
            except Exception:
                pass
        stages["research"] = round((time.time() - t0) * 1000)
        result = {"answer": report, "sources": sources, "citations": [], "related_questions": [], "cached": False, "tool_used": "research", "mode": "research"}
        answer_preview = result.get("answer", "")[:100]
        logger.info("v1 research_result answer_len=%d sources=%d", len(result.get("answer", "")), len(sources))
    else:
        logger.info("v1 route=rag: %.80s", message)
        result = orch.kb_query(message, user_id=user_id, kb_namespace=kb_namespace)
        stages["rag_query"] = round((time.time() - t0) * 1000)
        result["mode"] = "rag"
        answer_preview = result.get("answer", "")[:100]
        logger.info("v1 rag_result answer=%.100s sources=%d mode=%s", answer_preview, len(result.get("sources", [])), result.get("mode", "rag"))

    total_ms = round((time.time() - t_start) * 1000)
    logger.info("v1 latency stages=%s total=%dms tool=%s answer_len=%d", stages, total_ms, result.get("mode", "unknown"), len(result.get("answer", "")))
    result["_latency_ms"] = total_ms
    result["_stages"] = stages
    return result


@router.post("/v1/chat")
def v1_chat(req: V1ChatRequest, current_user: dict = Depends(get_current_user)):
    rid = uuid.uuid4().hex[:8]
    uid = current_user.get("id") if current_user else None
    ns = _resolve_kb_namespace(req.kb_id)
    message = _extract_message(req)
    _validate_message(message)
    mode = _determine_mode(message, req.mode)

    logger.info("[%s] v1/chat mode=%s msg=%.80s user=%s kb=%d conv=%s",
                rid, mode, message, uid, req.kb_id, req.conversation_id)
    t_start = time.time()

    try:
        result = _run_v1_chat(message, mode, uid, ns)
        total_ms = result.get("_latency_ms", round((time.time() - t_start) * 1000))

        answer = result.get("answer", "")
        logger.info("[%s] v1/chat response answer=%.120s mode=%s latency=%dms",
                    rid, answer[:120], result.get("mode", mode), total_ms)

        get_metrics().record("v1_chat", total_ms / 1000.0)
        return V1ChatResponse(
            answer=answer,
            mode=result.get("mode", mode),
            conversation_id=req.conversation_id,
            message_id=result.get("_msg_id", 0),
            sources=result.get("sources", []),
            citations=[CitationItem(**c) if isinstance(c, dict) else c for c in result.get("citations", [])],
            related_questions=result.get("related_questions", []),
            latency_ms=total_ms,
        )
    except (RAGError, LLMError) as e:
        get_metrics().record("v1_chat", time.time() - t_start, error=True)
        logger.error("[%s] v1/chat pipeline error: %s", rid, e)
        err = V1ErrorDetail(code="PIPELINE_ERROR", message=str(e))
        raise HTTPException(status_code=500, detail=err.model_dump())
    except Exception as e:
        get_metrics().record("v1_chat", time.time() - t_start, error=True)
        logger.error("[%s] v1/chat unexpected error: %s", rid, e)
        err = V1ErrorDetail(code="INTERNAL_ERROR", message="Pipeline execution failed")
        raise HTTPException(status_code=500, detail=err.model_dump())


@router.post("/v1/chat/stream")
def v1_chat_stream(req: V1ChatRequest, current_user: dict = Depends(get_current_user)):
    rid = uuid.uuid4().hex[:8]
    uid = current_user.get("id") if current_user else None
    ns = _resolve_kb_namespace(req.kb_id)
    message = _extract_message(req)
    _validate_message(message)
    mode = _determine_mode(message, req.mode)
    nc = req.conversation_id

    logger.info("[%s] v1/chat/stream mode=%s msg=%.80s user=%s kb=%d conv=%d",
                rid, mode, message, uid, req.kb_id, nc)

    def _stream():
        nonlocal nc
        t_start = time.time()
        answer = ""
        _mid = 0

        yield f"data: {json.dumps({'type': 'route', 'data': mode})}\n\n"
        yield f"data: {json.dumps({'type': 'debug', 'data': 'received_message', 'message': message[:200]})}\n\n"

        try:
            if mode == "live":
                logger.info("[%s] stream live_query for '%.60s'", rid, message)
                for event in _get_orch().live_stream(message, user_id=uid):
                    yield f"data: {event}\n\n"
                    try:
                        p = json.loads(event) if isinstance(event, str) else event
                        if p.get("type") == "done":
                            answer = p.get("data", "")
                    except Exception:
                        pass
                logger.info("[%s] stream live done answer_len=%d", rid, len(answer))

            elif mode == "research":
                logger.info("[%s] stream research for '%.60s'", rid, message)
                from ...research_agent import DeepResearchAgent
                agent = DeepResearchAgent()
                for event in agent.research(message, kb_namespace=ns):
                    yield f"data: {event}\n\n"
                    try:
                        p = json.loads(event) if isinstance(event, str) else event
                        if p["type"] == "research_token":
                            answer += p.get("token", "")
                        elif p["type"] == "research_done":
                            answer = p.get("report", answer)
                    except Exception:
                        pass
                logger.info("[%s] stream research done answer_len=%d", rid, len(answer))

            else:
                logger.info("[%s] stream rag_query for '%.60s'", rid, message)
                for chunk in _get_orch().stream_query(message, user_id=uid, kb_namespace=ns):
                    if chunk.startswith("__CITATIONS__"):
                        citations = chunk[len("__CITATIONS__"):]
                        yield f"data: {json.dumps({'type': 'citations', 'data': json.loads(citations)})}\n\n"
                        continue
                    answer += chunk
                    yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
                logger.info("[%s] stream rag done answer_len=%d answer=%.80s", rid, len(answer), answer)

        except Exception as e:
            logger.error("[%s] v1/chat/stream error: %s", rid, e)
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            return

        total_ms = round((time.time() - t_start) * 1000)
        logger.info("[%s] v1/chat/stream done total=%dms tool=%s answer_len=%d",
                    rid, total_ms, mode, len(answer))

        yield f"data: {json.dumps({'type': 'done', 'data': answer, 'latency_ms': total_ms})}\n\n"

        if uid and answer:
            try:
                from ...database import session_scope, ConversationRepository, MessageRepository
                with session_scope() as session:
                    crepo = ConversationRepository(session)
                    mrepo = MessageRepository(session)
                    if not nc:
                        conv = crepo.create(uid, title=message[:80])
                        nc = conv.id
                    mrepo.add(nc, "user", message, model=mode)
                    msg = mrepo.add(nc, "assistant", answer, live=(mode == "live"), model=mode)
                    _mid = msg.id
            except Exception as e:
                logger.warning("[%s] Failed to save conversation: %s", rid, e)

        if _mid:
            yield f"data: {json.dumps({'type': 'msg_id', 'data': _mid})}\n\n"
        if nc and nc != req.conversation_id:
            yield f"data: {json.dumps({'type': 'conv_id', 'data': nc})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
