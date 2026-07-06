import json
import time
import uuid
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse

from ...schemas.schemas import ChatRequest
from ...auth.middleware import get_current_user
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Research"])


def _get_research_agent():
    from ...research_agent import DeepResearchAgent
    agent = DeepResearchAgent()
    return agent


@router.post("/research")
def research_endpoint(
    req: ChatRequest,
    conv_id: int = Query(0),
    kb_id: int = Query(0),
    current_user: dict = Depends(get_current_user),
):
    uid = current_user.get("id") if current_user else None
    logger.info("Research request: %.80s (user=%s, conv=%s, kb_id=%d)", req.question, uid, conv_id, kb_id)

    agent = _get_research_agent()

    # Resolve kb_id to namespace
    kb_namespace = ""
    if kb_id:
        try:
            from ...database import session_scope, KnowledgeBaseRepository
            with session_scope() as session:
                repo = KnowledgeBaseRepository(session)
                kb = repo.get(kb_id)
                if kb:
                    kb_namespace = kb.collection_name
        except Exception:
            pass

    def _stream():
        rid = uuid.uuid4().hex[:8]
        answer = ""
        _mid = 0
        _nc = conv_id

        for event in agent.research(req.question, kb_namespace=kb_namespace):
            yield f"data: {event}\n\n"
            try:
                p = json.loads(event)
                if p["type"] == "research_token":
                    answer += p.get("token", "")
                elif p["type"] == "research_done":
                    answer = p.get("report", answer)
                elif p["type"] == "research_error":
                    answer = p.get("error", "")
            except Exception:
                pass

        # Save to conversation if user is authenticated and we got an answer
        if uid and answer:
            try:
                from ...database import session_scope, ConversationRepository, MessageRepository
                with session_scope() as session:
                    crepo = ConversationRepository(session)
                    mrepo = MessageRepository(session)
                    if not _nc:
                        conv = crepo.create(uid, title=req.question[:80])
                        _nc = conv.id
                    mrepo.add(_nc, "user", req.question, model="research")
                    msg = mrepo.add(_nc, "assistant", answer, live=False, model="research")
                    _mid = msg.id
            except Exception as e:
                logger.warning("Failed to save research conversation: %s", e)

        if _mid:
            yield f"data: {json.dumps({'type': 'research_msg_id', 'data': _mid})}\n\n"
        if _nc and _nc != conv_id:
            yield f"data: {json.dumps({'type': 'conv_id', 'data': _nc})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
