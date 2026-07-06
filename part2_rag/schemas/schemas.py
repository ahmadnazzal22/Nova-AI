from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "web"


class QueryResponse(BaseModel):
    answer: str
    sources: list[str | SourceItem] = []
    citations: list[CitationItem] = []
    related_questions: list[str] = []
    conversation_id: int = 0
    message_id: int = 0
    cached: bool = False
    tool_used: str | None = None


class LiveQueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = []
    citations: list[CitationItem] = []
    live: bool = True
    cached: bool = False
    related_questions: list[str] = []
    conversation_id: int = 0
    message_id: int = 0


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., max_length=120)
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=120)
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
    rating: int = Field(..., ge=1, le=5)
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
    title: str = Field(..., min_length=1, max_length=200)


class HealthResponse(BaseModel):
    status: str = "ok"
    documents_indexed: int = 0
    model_loaded: bool = False


class IngestResponse(BaseModel):
    filename: str
    chunks: int = 0
    total_docs: int = 0


class MemoryItem(BaseModel):
    id: int
    key: str
    value: str
    importance: float
    created_at: str


class MemoryStoreRequest(BaseModel):
    key: str
    value: str
    importance: float = 0.5


class CitationItem(BaseModel):
    id: int
    text: str = ""
    title: str = ""
    url: str = ""
    source: str = "kb"
    relevance_score: float = 0.0
    confidence_score: float = 0.0


class ChunkResult(BaseModel):
    text: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = "web"
    score: float = 0.0


class PipelineEvent(BaseModel):
    type: str
    data: Any = None
    message: str = ""


class V1ChatRequest(BaseModel):
    message: str = Field("", max_length=2000)
    question: str = Field("", max_length=2000)
    conversation_id: int = 0
    kb_id: int = 0
    mode: str = "auto"


class V1ChatResponse(BaseModel):
    answer: str
    mode: str
    conversation_id: int
    message_id: int
    sources: list = []
    citations: list[CitationItem] = []
    related_questions: list[str] = []
    latency_ms: int = 0


class V1ErrorDetail(BaseModel):
    code: str
    message: str
    details: list = []


class V1ErrorResponse(BaseModel):
    error: V1ErrorDetail
