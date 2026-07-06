from fastapi import APIRouter, Depends

from ...auth.middleware import require_admin
from ...database import get_session, UserRepository, StatsRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Admin"])


@router.get("/admin/users")
def admin_users(page: int = 1, per_page: int = 20, admin: dict = Depends(require_admin)):
    with get_session() as session:
        repo = UserRepository(session)
        users, total = repo.list(page, per_page)
        return {
            "total": total, "page": page, "per_page": per_page,
            "users": [{"id": u.id, "username": u.username, "email": u.email, "role": u.role, "is_active": u.is_active} for u in users],
        }


@router.get("/admin/stats")
def admin_stats(admin: dict = Depends(require_admin)):
    with get_session() as session:
        srepo = StatsRepository(session)
        return srepo.get_system_stats()


@router.get("/admin/export")
def admin_export(admin: dict = Depends(require_admin)):
    with get_session() as session:
        from ...models import User, Conversation, Message, Feedback
        users = [{"id": u.id, "username": u.username, "email": u.email, "role": u.role, "is_active": u.is_active} for u in session.query(User).all()]
        convs = [{"id": c.id, "user_id": c.user_id, "title": c.title} for c in session.query(Conversation).all()]
        msgs = [{"id": m.id, "conversation_id": m.conversation_id, "role": m.role, "content": m.content[:100]} for m in session.query(Message).all()]
        fb = [{"id": f.id, "message_id": f.message_id, "rating": f.rating} for f in session.query(Feedback).all()]
    return {"users": users, "conversations": convs, "messages": msgs, "feedback": fb}
