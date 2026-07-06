import os
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from part1_transformer.config import TransformerConfig
from .inference_server import InferenceServer
from .logger import get_logger

logger = get_logger(__name__)

config = TransformerConfig()

inference_server: InferenceServer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_server
    try:
        inference_server = InferenceServer(
            device=os.getenv("INFERENCE_DEVICE", "cuda"),
            use_fp16=os.getenv("INFERENCE_FP16", "true").lower() == "true",
            max_batch_size=int(os.getenv("INFERENCE_MAX_BATCH", "8")),
        )
        await inference_server.start()
        logger.info("API v2 ready")
    except Exception as e:
        logger.critical("Failed to start InferenceServer: %s", e)
        inference_server = None
    yield
    if inference_server:
        await inference_server.stop()


app = FastAPI(
    title="Transformer Inference API",
    description="High-performance GPU inference server with continuous batching",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Pydantic models ──────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(False)


class GenerateResponse(BaseModel):
    id: str
    text: str
    tokens_generated: int
    finish_reason: str


class ChatMessage(BaseModel):
    role: str = Field("user")
    content: str = Field(...)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=20)
    max_tokens: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(False)


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    gpu: dict
    scheduler: dict


# ── Helper ───────────────────────────────────────────────────

def _check_server():
    if inference_server is None:
        raise HTTPException(status_code=503, detail="Inference server not available")


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    _check_server()
    return inference_server.health()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    _check_server()
    request_id = uuid.uuid4().hex[:12]
    logger.info("[%s] generate | prompt=%.60s | max_tokens=%d | temp=%.2f",
                request_id, req.prompt, req.max_tokens, req.temperature)

    if req.stream:
        return StreamingResponse(
            _stream_generate(req.prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id},
        )

    text = await inference_server.generate(req.prompt, req.max_tokens, req.temperature)
    tokens = len(text.split())
    logger.info("[%s] done | tokens=%d | text=%.60s", request_id, tokens, text)
    return GenerateResponse(id=request_id, text=text, tokens_generated=tokens, finish_reason="stop")


@app.post("/chat")
async def chat(req: ChatRequest):
    _check_server()
    request_id = uuid.uuid4().hex[:12]
    prompt = "\n".join(f"{m.role}: {m.content}" for m in req.messages)
    logger.info("[%s] chat | messages=%d | max_tokens=%d", request_id, len(req.messages), req.max_tokens)

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id},
        )

    text = await inference_server.generate(prompt, req.max_tokens, req.temperature)
    return {
        "id": request_id,
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"total_tokens": len(text.split())},
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    _check_server()
    await websocket.accept()
    logger.info("WebSocket connected")

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 128)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            await websocket.send_json({"error": "prompt is required"})
            await websocket.close()
            return

        async for token in inference_server.generate_stream(prompt, max_tokens, temperature):
            await websocket.send_json({"type": "token", "content": token})

        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


# ── Streaming helpers ────────────────────────────────────────

async def _stream_generate(prompt: str, max_tokens: int, temperature: float):
    async for token in inference_server.generate_stream(prompt, max_tokens, temperature):
        yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat(prompt: str, max_tokens: int, temperature: float):
    async for token in inference_server.generate_stream(prompt, max_tokens, temperature):
        yield f"data: {json.dumps({'choices': [{'delta': {'content': token}}]})}\n\n"
    yield "data: [DONE]\n\n"
