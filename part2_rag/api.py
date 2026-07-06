import os
import json
import time
import uuid
import tempfile
import threading

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

from part1_transformer.config import TransformerConfig
from .rag_agent import RAGAgent
from .chunker import SUPPORTED_EXTS
from .exceptions import RAGError, LLMError
from .logger import get_logger
from .database import init_db, get_session, session_scope, \
    UserRepository, ConversationRepository, MessageRepository, FeedbackRepository, StatsRepository, MemoryRepository
from .auth import create_token_pair, refresh_access_token, hash_password, verify_password, \
    get_current_user, require_user, require_admin

logger = get_logger(__name__)
config = TransformerConfig()

rate_limit_store: dict[str, list[float]] = {}
_rate_limit_lock = threading.Lock()

app = FastAPI(
    title="Custom Transformer RAG API",
    description="Production-grade RAG system with authentication, chat history, feedback, and monitoring.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={"name": "RAG Platform Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","),
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=True)
    def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    @app.get("/", include_in_schema=True)
    def root_fallback():
        return {"status": "ok", "model_loaded": True}

_agent: RAGAgent | None = None
_agent_lock = threading.Lock()


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = RAGAgent(config=config)
    return _agent


def _agent_health_check():
    try:
        return get_agent()
    except Exception as e:
        logger.critical("RAGAgent init failed: %s", e)
        return None


# ── Rate Limiting ─────────────────────────────────────────────────

def _rate_limit(key: str, max_requests: int = 60, window: int = 60):
    now = time.time()
    with _rate_limit_lock:
        if key not in rate_limit_store:
            rate_limit_store[key] = []
        rate_limit_store[key] = [t for t in rate_limit_store[key] if now - t < window]
        if len(rate_limit_store[key]) >= max_requests:
            return False
        rate_limit_store[key].append(now)
    return True


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store"
    return response


# ── Lifespan ──────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Startup complete: DB initialized")


# ── Models ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is a transformer?"])


class SourceItem(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "web"


class QueryResponse(BaseModel):
    answer: str
    sources: list[str | SourceItem] = []
    related_questions: list[str] = []
    conversation_id: int = 0
    message_id: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int = 0
    model_loaded: bool = False


class IngestResponse(BaseModel):
    filename: str
    chunks: int = 0
    total_docs: int = 0


class LiveQueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    live: bool = True
    cached: bool = False
    related_questions: list[str] = []
    conversation_id: int = 0
    message_id: int = 0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["What is attention in deep learning?"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, examples=["johndoe"])
    email: str = Field(..., max_length=120, examples=["john@example.com"])
    password: str = Field(..., min_length=6, max_length=128, examples=["securepass123"])


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=120, examples=["john@example.com"])
    password: str = Field(..., max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    role: str
    last_login: str | None = None
    created_at: str | None = None


class UpdateProfileRequest(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=50)
    email: str | None = Field(None, max_length=120)


class UpdateSettingsRequest(BaseModel):
    theme: str | None = None
    default_mode: str | None = None
    default_sources: int | None = None
    streaming_enabled: bool | None = None
    preferences: dict | None = None


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    comment: str = ""


class ConversationListItem(BaseModel):
    id: int
    title: str
    msg_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    sources: list = []
    related_questions: list = []
    live: bool = False
    cached: bool = False
    model: str = ""
    response_time: float = 0
    created_at: str | None = None


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, examples=["My renamed conversation"])


# ── Helper: save conversation ─────────────────────────────────────

def _save_conv(uid: int | None, conv_id: int, question: str, result: dict, elapsed: float, model: str = "live") -> int:
    if not uid:
        return conv_id
    try:
        with session_scope() as session:
            crepo = ConversationRepository(session)
            mrepo = MessageRepository(session)
            if not conv_id:
                conv = crepo.create(uid, title=question[:80])
                conv_id = conv.id
            mrepo.add(conv_id, "user", question, model=model)
            msg = mrepo.add(
                conv_id, "assistant", result.get("answer", ""),
                sources=result.get("sources", []),
                related_questions=result.get("related_questions", []),
                live=result.get("live", False),
                cached=result.get("cached", False),
                response_time=elapsed,
                model=model,
            )
            result["_msg_id"] = msg.id
    except Exception as e:
        logger.warning("Failed to save conversation: %s", e)
    return conv_id


# ── Endpoints: Health ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503, detail="RAGAgent not initialized")
    try:
        count = agent.db._collection.count()
    except Exception:
        count = 0
    return HealthResponse(status="ok", documents_indexed=count, model_loaded=True)


# ── Endpoints: Query ──────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_endpoint(req: QueryRequest, conv_id: int = 0, current_user: dict = Depends(get_current_user)):
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503, detail="RAGAgent not initialized")
    if not _rate_limit(f"query:{current_user.get('id', 'anon') if current_user else 'anon'}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rid = uuid.uuid4().hex[:8]
    logger.info("[%s] Query: %.80s", rid, req.question)
    try:
        uid = current_user.get("id") if current_user else None
        result = agent.query(req.question, user_id=uid)
        conv_id = _save_conv(uid, conv_id, req.question, result, 0, model="kb")
        return QueryResponse(
            answer=result["answer"], sources=result.get("sources", []),
            related_questions=result.get("related_questions", []),
            conversation_id=conv_id, message_id=result.get("_msg_id", 0),
        )
    except (RAGError, LLMError) as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["RAG"])
async def ingest_endpoint(
    file: UploadFile = File(..., description=".txt, .md, .pdf"),
    chunk_size: int = Form(512, ge=64, le=4096),
    chunk_overlap: int = Form(64, ge=0, le=512),
):
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503)
    fid = uuid.uuid4().hex[:8]
    fn = file.filename or "unknown"
    ext = os.path.splitext(fn)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported type '{ext}'. Supported: {', '.join(SUPPORTED_EXTS)}")
    tmp = None
    try:
        data = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp.write(data); tmp.close()
        logger.info("[%s] Ingest: %s (%d B)", fid, fn, len(data))
        stats = agent.ingest_file(tmp.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        stats["filename"] = fn
        return IngestResponse(**stats)
    except RAGError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


def _stream_answer(question: str, uid: int | None = None):
    for chunk in get_agent().stream_query(question, user_id=uid):
        yield f"data: {chunk}\n\n"


@app.post("/query/stream", tags=["RAG"])
def query_stream_endpoint(req: QueryRequest):
    get_agent()
    return StreamingResponse(_stream_answer(req.question), media_type="text/event-stream")


@app.post("/chat", response_model=QueryResponse, tags=["RAG"])
def chat_endpoint(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    return query_endpoint(QueryRequest(question=req.question), current_user=current_user)


def _stream_chat_full(question: str, uid: int | None, conv_id: int):
    agent = get_agent()
    t0 = time.time()
    tokens: list[str] = []
    for chunk in agent.stream_query(question, user_id=uid):
        tokens.append(chunk)
        yield f"data: {json.dumps({'token': chunk})}\n\n"
    answer = "".join(tokens)
    elapsed = time.time() - t0
    # Build result dict for saving
    result = {"answer": answer, "sources": [], "related_questions": [], "live": False, "cached": False}
    if uid:
        try:
            with session_scope() as session:
                crepo = ConversationRepository(session)
                mrepo = MessageRepository(session)
                new_conv_id = conv_id
                if not new_conv_id:
                    conv = crepo.create(uid, title=question[:80])
                    new_conv_id = conv.id
                mrepo.add(new_conv_id, "user", question, model="rag")
                msg = mrepo.add(new_conv_id, "assistant", answer, response_time=elapsed, model="rag")
                result["_msg_id"] = msg.id
                conv_id = new_conv_id
        except Exception as e:
            logger.warning("Failed to save streaming conversation: %s", e)
    yield f"data: {json.dumps({'done': True, 'conv_id': conv_id, 'msg_id': result.get('_msg_id', 0)})}\n\n"


@app.post("/chat/stream", tags=["RAG"])
def chat_stream_endpoint(req: ChatRequest, conv_id: int = 0, current_user: dict = Depends(get_current_user)):
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503)
    if not _rate_limit(f"chat:{current_user.get('id', 'anon') if current_user else 'anon'}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    uid = current_user.get("id") if current_user else None
    return StreamingResponse(
        _stream_chat_full(req.question, uid, conv_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/live", response_model=LiveQueryResponse, tags=["RAG"])
def live_endpoint(req: ChatRequest, conv_id: int = 0, current_user: dict = Depends(get_current_user)):
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503)
    if not _rate_limit(f"live:{current_user.get('id', 'anon') if current_user else 'anon'}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    rid = uuid.uuid4().hex[:8]
    logger.info("[%s] Live: %.80s", rid, req.question)
    t0 = time.time()
    uid = current_user.get("id") if current_user else None
    try:
        result = agent.query_live(req.question, user_id=uid)
        elapsed = time.time() - t0
        conv_id = _save_conv(uid, conv_id, req.question, result, elapsed, model="live")
        return LiveQueryResponse(
            answer=result.get("answer", ""), sources=result.get("sources", []),
            live=result.get("live", True), cached=result.get("cached", False),
            related_questions=result.get("related_questions", []),
            conversation_id=conv_id, message_id=result.get("_msg_id", 0),
        )
    except (RAGError, LLMError) as e:
        raise HTTPException(status_code=500, detail=str(e))


def _stream_live(question: str, uid: int | None = None, conv_id: int = 0):
    agent = get_agent()
    t0 = time.time()
    tokens: list[str] = []
    answer = ""
    sources = []
    related = []
    for line in agent.query_live_stream(question, user_id=uid):
        yield f"data: {line}\n\n"
        try:
            parsed = json.loads(line)
            if parsed["type"] == "token":
                tokens.append(parsed["data"])
            elif parsed["type"] == "done":
                answer = parsed["data"]
            elif parsed["type"] == "sources":
                sources = parsed.get("data", [])
            elif parsed["type"] == "related":
                related = parsed.get("data", [])
        except Exception:
            pass
    elapsed = time.time() - t0
    if uid and answer:
        try:
            with session_scope() as session:
                crepo = ConversationRepository(session)
                mrepo = MessageRepository(session)
                new_conv_id = conv_id
                if not new_conv_id:
                    conv = crepo.create(uid, title=question[:80])
                    new_conv_id = conv.id
                mrepo.add(new_conv_id, "user", question, model="live")
                msg = mrepo.add(new_conv_id, "assistant", answer,
                    sources=sources, related_questions=related,
                    live=True, response_time=elapsed, model="live")
                yield f"data: {json.dumps({'type': 'conv_id', 'data': new_conv_id})}\n\n"
                yield f"data: {json.dumps({'type': 'msg_id', 'data': msg.id})}\n\n"
        except Exception as e:
            logger.warning("Failed to save live stream: %s", e)


@app.post("/live/stream", tags=["RAG"])
def live_stream_endpoint(req: ChatRequest, conv_id: int = 0, current_user: dict = Depends(get_current_user)):
    agent = _agent_health_check()
    if agent is None:
        raise HTTPException(status_code=503)
    if not _rate_limit(f"live:{current_user.get('id', 'anon') if current_user else 'anon'}"):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    uid = current_user.get("id") if current_user else None
    return StreamingResponse(
        _stream_live(req.question, uid=uid, conv_id=conv_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Auth Endpoints ────────────────────────────────────────────────

@app.post("/auth/register", response_model=TokenResponse, tags=["Auth"])
def register(req: RegisterRequest):
    with session_scope() as session:
        repo = UserRepository(session)
        if repo.get_by_username(req.username):
            raise HTTPException(400, "Username already exists")
        if repo.get_by_email(req.email):
            raise HTTPException(400, "Email already exists")
        user = repo.create(req.username, req.email, hash_password(req.password))
        if not user:
            raise HTTPException(500, "Failed to create user")
        uid, uname, urole = user.id, user.username, user.role
    tokens = create_token_pair(uid, uname, urole)
    return TokenResponse(**tokens)


@app.post("/auth/login", response_model=TokenResponse, tags=["Auth"])
def login(req: LoginRequest):
    with session_scope() as session:
        repo = UserRepository(session)
        user = repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(401, "Invalid email or password")
        if not user.is_active:
            raise HTTPException(403, "Account deactivated")
        repo.update_login(user.id)
        uid, uname, urole = user.id, user.username, user.role
    tokens = create_token_pair(uid, uname, urole)
    return TokenResponse(**tokens)


@app.post("/auth/refresh", response_model=TokenResponse, tags=["Auth"])
def refresh(req: RefreshRequest):
    tokens = refresh_access_token(req.refresh_token)
    if not tokens:
        raise HTTPException(401, "Invalid or expired refresh token")
    return TokenResponse(**tokens)


@app.get("/auth/me", response_model=UserProfile, tags=["Auth"])
def auth_me(current_user: dict = Depends(require_user)):
    with get_session() as session:
        repo = UserRepository(session)
        user = repo.get_by_id(current_user["id"])
        if not user:
            raise HTTPException(404, "User not found")
        profile = UserProfile(
            id=user.id, username=user.username, email=user.email, role=user.role,
            last_login=str(user.last_login) if user.last_login else None,
            created_at=str(user.created_at) if user.created_at else None,
        )
    return profile


# ── User Management ───────────────────────────────────────────────

@app.put("/auth/profile", response_model=UserProfile, tags=["Users"])
def update_profile(req: UpdateProfileRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        updates = {}
        if req.username:
            existing = repo.get_by_username(req.username)
            if existing and existing.id != current_user["id"]:
                raise HTTPException(400, "Username taken")
            updates["username"] = req.username
        if req.email:
            existing = repo.get_by_email(req.email)
            if existing and existing.id != current_user["id"]:
                raise HTTPException(400, "Email taken")
            updates["email"] = req.email
        user = repo.update_profile(current_user["id"], **updates)
        if not user:
            raise HTTPException(404, "User not found")
        profile = UserProfile(
            id=user.id, username=user.username, email=user.email, role=user.role,
            last_login=str(user.last_login) if user.last_login else None,
            created_at=str(user.created_at) if user.created_at else None,
        )
    return profile


@app.get("/auth/settings", tags=["Users"])
def get_settings(current_user: dict = Depends(require_user)):
    with get_session() as session:
        repo = UserRepository(session)
        settings = repo.get_settings(current_user["id"])
        if not settings:
            return {}
        result = {
            "theme": settings.theme,
            "default_mode": settings.default_mode,
            "default_sources": settings.default_sources,
            "streaming_enabled": settings.streaming_enabled,
            "preferences": settings.preferences or {},
        }
    return result


@app.put("/auth/settings", tags=["Users"])
def update_settings(req: UpdateSettingsRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
        settings = repo.update_settings(current_user["id"], **updates)
    if not settings:
        raise HTTPException(404, "Settings not found")
    return {"status": "updated"}


@app.delete("/auth/account", tags=["Users"])
def delete_account(current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        if not repo.deactivate(current_user["id"]):
            raise HTTPException(404, "User not found")
    return {"status": "account deactivated"}


# ── Chat History ──────────────────────────────────────────────────

@app.get("/chat/history", response_model=list[ConversationListItem], tags=["Chat"])
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


@app.get("/chat/{conv_id}", tags=["Chat"])
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
        cid, ctitle = conv.id, conv.title
    return {"conversation": {"id": cid, "title": ctitle}, "messages": msg_items}


@app.put("/chat/{conv_id}/rename", tags=["Chat"])
def chat_rename(conv_id: int, req: RenameRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        crepo = ConversationRepository(session)
        conv = crepo.get_by_id(conv_id)
        if not conv or conv.user_id != current_user["id"]:
            raise HTTPException(404, "Conversation not found")
        crepo.rename(conv_id, req.title)
    return {"status": "renamed", "title": req.title}


@app.delete("/chat/{conv_id}", tags=["Chat"])
def chat_delete(conv_id: int, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        crepo = ConversationRepository(session)
        conv = crepo.get_by_id(conv_id)
        if not conv or conv.user_id != current_user["id"]:
            raise HTTPException(404, "Conversation not found")
        crepo.delete(conv_id)
    return {"status": "deleted"}


# ── Feedback ──────────────────────────────────────────────────────

@app.post("/feedback", tags=["Feedback"])
def submit_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    uid = current_user.get("id") if current_user else None
    with session_scope() as session:
        frepo = FeedbackRepository(session)
        fb = frepo.add(req.message_id, uid, req.rating, req.comment)
        if not fb:
            raise HTTPException(400, "Failed to submit feedback")
        fb_id = fb.id
    return {"status": "ok", "feedback_id": fb_id}


@app.get("/feedback/stats", tags=["Feedback"])
def feedback_stats(current_user: dict = Depends(require_user)):
    with get_session() as session:
        frepo = FeedbackRepository(session)
        return frepo.get_stats()


# ── Memory ────────────────────────────────────────────────────────

class MemoryResponse(BaseModel):
    id: int
    key: str
    value: str
    importance: float
    created_at: str

class MemoryStoreRequest(BaseModel):
    key: str
    value: str
    importance: float = 0.5

@app.get("/memories", tags=["Memory"])
def list_memories(limit: int = 20, user: dict = Depends(require_user)):
    with get_session() as session:
        repo = MemoryRepository(session)
        memories = repo.get_recent(user["id"], limit=limit)
        return {"memories": [
            {"id": m.id, "key": m.key, "value": m.value, "importance": m.importance,
             "created_at": m.created_at.isoformat()}
            for m in memories
        ]}

@app.post("/memories", tags=["Memory"])
def store_memory(req: MemoryStoreRequest, user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = MemoryRepository(session)
        mem = repo.store(user["id"], req.key, req.value, req.importance)
        return {"id": mem.id, "key": mem.key, "value": mem.value, "importance": mem.importance}

@app.delete("/memories/{key}", tags=["Memory"])
def delete_memory(key: str, user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = MemoryRepository(session)
        if repo.delete_by_key(user["id"], key):
            return {"deleted": key}
        raise HTTPException(status_code=404, detail="Memory not found")

@app.get("/memories/search", tags=["Memory"])
def search_memories(q: str, user: dict = Depends(require_user)):
    with get_session() as session:
        repo = MemoryRepository(session)
        results = repo.search(user["id"], q)
        return {"results": [
            {"id": m.id, "key": m.key, "value": m.value, "importance": m.importance,
             "created_at": m.created_at.isoformat()}
            for m in results
        ]}


# ── Admin ─────────────────────────────────────────────────────────

@app.get("/admin/users", tags=["Admin"])
def admin_users(page: int = 1, per_page: int = 20, admin: dict = Depends(require_admin)):
    with get_session() as session:
        repo = UserRepository(session)
        users, total = repo.list(page, per_page)
        return {
            "total": total,
            "page": page,
            "per_page": per_page,
            "users": [{"id": u.id, "username": u.username, "email": u.email, "role": u.role, "is_active": u.is_active} for u in users],
        }


@app.get("/admin/stats", tags=["Admin"])
def admin_stats(admin: dict = Depends(require_admin)):
    with get_session() as session:
        srepo = StatsRepository(session)
        return srepo.get_system_stats()


@app.get("/admin/export", tags=["Admin"])
def admin_export(admin: dict = Depends(require_admin)):
    with get_session() as session:
        from .models import User, Conversation, Message, Feedback
        users = [{"id": u.id, "username": u.username, "email": u.email, "role": u.role, "is_active": u.is_active}
                 for u in session.query(User).all()]
        convs = [{"id": c.id, "user_id": c.user_id, "title": c.title} for c in session.query(Conversation).all()]
        msgs = [{"id": m.id, "conversation_id": m.conversation_id, "role": m.role, "content": m.content[:100]}
                for m in session.query(Message).all()]
        fb = [{"id": f.id, "message_id": f.message_id, "rating": f.rating} for f in session.query(Feedback).all()]
    return {"users": users, "conversations": convs, "messages": msgs, "feedback": fb}
