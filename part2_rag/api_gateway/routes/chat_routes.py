from fastapi import APIRouter, HTTPException, Depends

from ...schemas.schemas import ConversationListItem, MessageItem, RenameRequest
from ...auth.middleware import require_user
from ...database import get_session, session_scope, ConversationRepository, MessageRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Chat"])


@router.get("/chat/history", response_model=list[ConversationListItem])
def chat_history(current_user: dict = Depends(require_user)):
    with get_session() as session:
        crepo = ConversationRepository(session)
        mrepo = MessageRepository(session)
        convs = crepo.list_by_user(current_user["id"])
        items = []
        for c in convs:
            msgs = mrepo.list_by_conversation(c.id)
            items.append(ConversationListItem(
                id=c.id, title=c.title, msg_count=len(msgs),
                created_at=str(c.created_at) if c.created_at else None,
                updated_at=str(c.updated_at) if c.updated_at else None,
            ))
    return items


@router.get("/chat/{conv_id}")
def chat_conversation(conv_id: int, current_user: dict = Depends(require_user)):
    with get_session() as session:
        crepo = ConversationRepository(session)
        mrepo = MessageRepository(session)
        conv = crepo.get_by_id(conv_id)
        if not conv or conv.user_id != current_user["id"]:
            raise HTTPException(404, "Conversation not found")
        msgs = mrepo.list_by_conversation(conv_id)
        msg_items = []
        for m in msgs:
            msg_items.append(MessageItem(
                id=m.id, role=m.role, content=m.content,
                sources=m.sources or [], related_questions=m.related_questions or [],
                live=m.live, cached=m.cached, model=m.model, response_time=m.response_time,
                created_at=str(m.created_at) if m.created_at else None,
            ))
    return {"conversation": {"id": conv.id, "title": conv.title}, "messages": msg_items}


@router.put("/chat/{conv_id}/rename")
def chat_rename(conv_id: int, req: RenameRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        crepo = ConversationRepository(session)
        conv = crepo.get_by_id(conv_id)
        if not conv or conv.user_id != current_user["id"]:
            raise HTTPException(404, "Conversation not found")
        crepo.rename(conv_id, req.title)
    return {"status": "renamed", "title": req.title}


@router.delete("/chat/{conv_id}")
def chat_delete(conv_id: int, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        crepo = ConversationRepository(session)
        conv = crepo.get_by_id(conv_id)
        if not conv or conv.user_id != current_user["id"]:
            raise HTTPException(404, "Conversation not found")
        crepo.delete(conv_id)
    return {"status": "deleted"}
