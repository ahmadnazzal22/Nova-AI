"""
Production API Gateway — FastAPI application that routes to microservices.
V1 unified API with clean error handling and no backward compatibility.
"""
import os
import time

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError

from ..config.settings import settings
from ..logger import get_logger
from ..database import init_db

from ..auth.middleware import get_current_user, require_user

from .routes.auth_routes import router as auth_router
from .routes.chat_routes import router as chat_router
from .routes.feedback_routes import router as feedback_router
from .routes.ingest_routes import router as ingest_router
from .routes.memory_routes import router as memory_router
from .routes.admin_routes import router as admin_router
from .routes.kb_routes import router as kb_router
from .routes.v1_routes import router as v1_router

logger = get_logger(__name__)

gateway_app = FastAPI(
    title=settings.app_name,
    description="V1 RAG API Gateway — unified chat contract",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

gateway_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gateway_app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts.split(","),
)


@gateway_app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cache-Control"] = "no-store"
    return response


@gateway_app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    path = request.url.path
    method = request.method
    logger.info("→ %s %s", method, path)
    try:
        response = await call_next(request)
        elapsed = round((time.time() - t0) * 1000)
        logger.info("← %s %s → %s (%dms)", method, path, response.status_code, elapsed)
        if elapsed > 5000:
            logger.warning("SLOW (%dms): %s %s", elapsed, method, path)
        return response
    except Exception as e:
        elapsed = round((time.time() - t0) * 1000)
        logger.error("✗ %s %s → ERROR (%dms): %s", method, path, elapsed, e)
        raise


@gateway_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


@gateway_app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                "details": [],
            }
        },
    )


@gateway_app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "not_found",
                "message": f"The requested resource was not found: {request.url.path}",
                "details": [],
            }
        },
    )


@gateway_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "details": [],
            }
        },
    )


@gateway_app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Gateway startup complete")


static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
if os.path.isdir(static_dir):
    gateway_app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @gateway_app.get("/", include_in_schema=True)
    def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))
else:
    @gateway_app.get("/", include_in_schema=True)
    def root_fallback():
        return {"status": "ok", "model_loaded": True}


@gateway_app.get("/health", tags=["System"])
def health():
    from ..monitoring.metrics import get_metrics
    metrics = get_metrics()
    return {"status": "ok", "documents_indexed": 0, "model_loaded": True}


@gateway_app.get("/sources/{message_id}", tags=["System"])
def get_message_sources(message_id: int, current_user: dict = Depends(get_current_user)):
    import json
    try:
        from ..database import session_scope, MessageRepository
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


gateway_app.include_router(v1_router)
gateway_app.include_router(auth_router)
gateway_app.include_router(chat_router)
gateway_app.include_router(feedback_router)
gateway_app.include_router(ingest_router)
gateway_app.include_router(memory_router)
gateway_app.include_router(admin_router)
gateway_app.include_router(kb_router)


@gateway_app.get("/metrics", tags=["Monitoring"])
def metrics_endpoint():
    from ..monitoring.metrics import get_metrics
    return get_metrics().get_metrics()
