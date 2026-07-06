from fastapi import APIRouter, HTTPException, Depends

from ...schemas.schemas import FeedbackRequest
from ...auth.middleware import get_current_user, require_user
from ...database import get_session, session_scope, FeedbackRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Feedback"])


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    with session_scope() as session:
        frepo = FeedbackRepository(session)
        fb = frepo.add(req.message_id, uid, req.rating, req.comment)
        if not fb:
            raise HTTPException(400, "Failed to submit feedback")
        fb_id = fb.id
    return {"status": "ok", "feedback_id": fb_id}


@router.get("/feedback/stats")
def feedback_stats(current_user: dict = Depends(require_user)):
    with get_session() as session:
        frepo = FeedbackRepository(session)
        return frepo.get_stats()
