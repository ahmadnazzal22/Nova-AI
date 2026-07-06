# Custom Transformer RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat&logo=pytorch">
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat">
  <img src="https://img.shields.io/badge/Tests-100%2B-brightgreen?style=flat">
  <img src="https://img.shields.io/badge/status-production-blue?style=flat">
</p>

<p align="center">
  A production-grade <strong>Retrieval-Augmented Generation</strong> system that builds a <strong>Transformer from scratch in PyTorch</strong> and connects it to an <strong>autonomous multi-agent RAG pipeline</strong> — where your own Transformer generates the embeddings for vector search.
</p>

---

## Features

- **From-scratch Transformer** — Multi-head attention, positional encoding, encoder-decoder, all in raw PyTorch (no HuggingFace)
- **Multi-route Query Engine** — Auto-routes to RAG (knowledge base), Live (web search), or Research (deep multi-step analysis)
- **Hybrid Retrieval** — Dense vector search (ChromaDB/Qdrant) + BM25 keyword search with reranking
- **Multi-LLM Support** — Groq API (primary), Ollama (local fallback), with circuit breaker & auto model selection
- **Streaming Responses** — SSE-based real-time token streaming
- **Deep Research Agent** — Decomposes complex questions into sub-questions, searches each, synthesizes a structured report
- **Live Web Search** — DuckDuckGo + Wikipedia backends with domain filtering
- **User Auth & Memory** — JWT authentication, conversation history, long-term memory extraction
- **Multi-KB Support** — Isolated knowledge bases with per-user permissions (read/write/admin)
- **Full Monitoring** — OpenTelemetry tracing, query latency metrics, cache hit rates
- **Production Ready** — Docker Compose, PostgreSQL, Redis, Qdrant, MinIO, Celery, Nginx

---

## Architecture

```
                        ┌──────────────────────────────┐
                        │     part1_transformer         │
                        │  Tokenizer → Transformer     │
                        │  (Pure PyTorch, no HF)       │
                        └──────────┬───────────────────┘
                                   │ Embeddings (mean/max/cls pooling)
                        ┌──────────▼───────────────────┐
                        │       part2_rag               │
                        │                               │
                        │  ┌───────────────────────┐    │
            ┌───────────┤  │   API Gateway (8002)   │    │
            │           │  │  /v1/chat  /v1/chat/s  │    │
            │           │  └───────────┬───────────┘    │
            │           │              │                 │
            │           │  ┌───────────▼───────────┐    │
            │           │  │   RAGOrchestrator      │    │
            │           │  │  ┌──────┬──────┬────┐ │    │
            │           │  │  │ RAG  │ Live │ Res│ │    │
            │           │  │  └──────┴──────┴────┘ │    │
            │           │  └───────────┬───────────┘    │
            │           │              │                 │
            │           │  ┌───────────▼───────────┐    │
            │           │  │     LLM Service       │    │
            │           │  │  Groq (llama-3.1-8b) │    │
            │           │  │  Ollama (phi3, etc)  │    │
            │           │  └─────────────────────┘    │
            │           │                              │
User ───────┤  HTTP     │  ┌──────┐ ┌────┐ ┌──────┐  │
            │  / SSE    │  │Qdrant│ │BM25│ │Redis │  │
            │           │  └──────┘ └────┘ └──────┘  │
            │           └──────────────────────────────┘
            │
            ├──→  Legacy API (8000)
            └──→  V2 Inference API
```

### Pipeline Flow

```
User Query
     │
     ▼
┌──────────────┐
│  Fast Path   │ ← Greeting? (hello, hi, thanks, bye)
└──────┬───────┘
       │
┌──────▼───────┐
│ Query Intel  │ ← Rewrite, intent, keywords, entities
└──────┬───────┘
       │
  ┌────┼────────────┐
  │    │            │
  ▼    ▼            ▼
┌────┐ ┌────┐  ┌────────┐
│RAG │ │Live│  │Research│
│    │ │    │  │        │
│Qd- │ │DDG │  │Sub-Q   │
│rant│ │+   │  │search  │
│+   │ │Wiki│  │+ synth │
│BM25│ │+LLM│  │+ report│
│+LLM│ │    │  │        │
└─┬──┘ └─┬──┘  └───┬────┘
  │      │         │
  └──────┼─────────┘
         │
    ┌────▼──────┐
    │LLM Service│
    │Groq/ Ollam│
    └────┬──────┘
         │
    ┌────▼──────┐
    │ Formatter │
    │ + Validat │
    └────┬──────┘
         │
    ┌────▼──────┐
    │ Response  │
    └───────────┘
```

---

## Project Structure

```
NEW DAY 18/
│
├── part1_transformer/            # Transformer built from scratch
│   ├── attention.py              # Multi-head self/cross attention + KV cache
│   ├── transformer.py            # Full encoder-decoder architecture
│   ├── tokenizer.py              # WordTokenizer (word + BPE modes)
│   ├── embeddings.py             # TokenEmbedding with d_model^0.5 scaling
│   ├── positional_encoding.py    # Sin/cos positional encodings
│   ├── feed_forward.py           # FFN: Linear → ReLU → Dropout → Linear
│   ├── config.py                 # .env-backed TransformerConfig
│   ├── dataset.py                # File-based loader + train/val split
│   ├── train.py                  # Training loop, checkpointing, embedding extraction
│   ├── inference_engine.py       # Async inference with continuous batching
│   ├── exceptions.py             # Custom exception hierarchy
│   └── logger.py                 # Structured logging
│
├── part2_rag/                    # Agentic RAG system
│   ├── api_gateway/              # Production API Gateway
│   │   ├── gateway.py            # Main FastAPI app (port 8002)
│   │   └── routes/
│   │       ├── v1_routes.py      # Unified /v1/chat contract (main entry)
│   │       ├── auth_routes.py    # Auth: register, login, refresh, me
│   │       ├── chat_routes.py    # Conversation history CRUD
│   │       ├── feedback_routes.py# Message rating/feedback
│   │       ├── ingest_routes.py  # Document upload & chunking
│   │       ├── memory_routes.py  # Long-term user memory
│   │       ├── admin_routes.py   # Admin: users, stats
│   │       ├── kb_routes.py      # Multi-knowledge-base management
│   │       ├── _legacy_rag_routes.py
│   │       └── _legacy_research_routes.py
│   │
│   ├── orchestrator/             # Orchestration layer
│   │   ├── rag_orchestrator.py   # Central pipeline coordinator
│   │   └── query_router.py       # Query → route classifier
│   │
│   ├── llm/                      # LLM integration
│   │   ├── llm_service.py        # Circuit breaker, model selection, MockLLM
│   │   └── groq_llm.py           # Groq API LangChain wrapper
│   │
│   ├── retrieval/                # Vector retrieval
│   │   ├── qdrant_store.py       # Qdrant adapter
│   │   ├── hybrid_retriever.py   # Dense + BM25 hybrid
│   │   └── bm25.py               # Keyword retriever
│   │
│   ├── ranking/                  # Ranking & reranking
│   │   └── reranker.py           # Keyword overlap, title relevance, noise penalty
│   │
│   ├── auth/                     # Authentication
│   │   ├── jwt_handler.py        # HMAC-SHA256 JWT
│   │   └── middleware.py         # FastAPI Depends
│   │
│   ├── schemas/                  # Pydantic v2 models
│   │   └── schemas.py            # V1ChatRequest, CitationItem, etc.
│   │
│   ├── cache/redis_cache.py      # Redis query cache
│   ├── context/memory_service.py # Short-term conversation memory
│   ├── citation/                 # Citation formatting & source resolution
│   ├── monitoring/               # OpenTelemetry + custom metrics
│   ├── storage/object_store.py   # MinIO S3 adapter
│   ├── worker/                   # Celery background tasks
│   │
│   ├── rag_agent.py              # Singleton RAG agent
│   ├── research_agent.py         # Deep multi-step research agent
│   ├── web_search.py             # DuckDuckGo + Wikipedia search
│   ├── web_loader.py             # Web page fetcher & chunker
│   ├── prompt_templates.py       # Intent-specific prompt templates
│   ├── prompt_router.py          # Intent → prompt mapper
│   ├── query_intelligence.py     # Query rewriting, intent, multi-query
│   ├── fast_path.py              # Greeting detection
│   ├── chunker.py                # Document chunking (txt, md, pdf)
│   ├── custom_embeddings.py      # TransformerEmbeddings LangChain wrapper
│   ├── response_formatter.py     # Markdown formatting
│   ├── response_validator.py     # Quality validation & fallback
│   ├── context_compressor.py     # Extractive + LLM-based compression
│   ├── tool_decision_llm.py      # LLM-based tool selection
│   ├── tool_router.py            # Tool routing
│   ├── intent_detector.py        # Rule-based intent detection
│   ├── memory_manager.py         # Long-term memory extraction
│   ├── inference_server.py       # Async GPU inference server
│   ├── database.py               # SQLAlchemy repos (User, Conv, Msg, etc.)
│   ├── models.py                 # SQLAlchemy ORM models
│   ├── exceptions.py             # Custom exception hierarchy
│   └── logger.py                 # Structured logging with correlation IDs
│
├── tests/                        # 100+ tests across 16 files
│   ├── test_tokenizer.py
│   ├── test_embeddings.py
│   ├── test_attention.py
│   ├── test_transformer.py
│   ├── test_train.py
│   ├── test_custom_embeddings.py
│   ├── test_inference_engine.py
│   ├── test_rag_agent.py
│   ├── test_chunker.py
│   ├── test_api.py
│   ├── test_api_integration.py
│   ├── test_auth.py
│   ├── test_database.py
│   ├── test_query_router.py
│   └── test_retrieval_quality.py
│
├── static/index.html             # SPA frontend (2276 lines)
├── alembic/                      # DB migrations (3 versions)
├── chroma_db/                    # Persisted ChromaDB vector store
├── nginx/nginx.conf              # Reverse proxy config
├── scripts/                      # Utilities
├── data/                         # Training data & ingested docs
├── samples/                      # Example API payloads
├── rag.db                        # SQLite database (dev)
│
├── run_gateway.py                # Start the API Gateway
├── run_api.py                    # Start legacy API on 8002
├── run_legacy.py                 # Start legacy API on 8000
├── run_celery_worker.py          # Start Celery worker
│
├── docker-compose.yml            # Full production stack
├── Dockerfile / Dockerfile.gateway
├── requirements.txt
├── .env.example                  # All config keys with defaults
└── pytest.ini                    # asyncio_mode = auto
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch (CPU is fine)
- Optional: [Ollama](https://ollama.ai) for local LLMs, [Groq API key](https://console.groq.com) for cloud LLM, [Redis](https://redis.io) for caching

### Setup

```bash
git clone <repo-url> && cd NEW-DAY-18
cp .env.example .env
pip install -r requirements.txt
```

Edit `.env` to add your `GROQ_API_KEY` or point to your Ollama server.

### Train the Transformer

```bash
python -m part1_transformer.train
```

Outputs: `transformer_best.pth`, `transformer_checkpoint.pth`, `embeddings.pth`

### Run the Server

```bash
# API Gateway (recommended — all features)
python run_gateway.py
# → http://localhost:8002

# Legacy API (basic RAG)
python run_legacy.py
# → http://localhost:8000
```

### Run with Docker

```bash
docker-compose up
# Gateway:8002  Postgres  Redis  Qdrant  MinIO  Flower:5555
```

### Run Tests

```bash
pytest tests/ -v
pytest tests/ --cov=part1_transformer --cov=part2_rag
```

---

## API Reference

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves frontend SPA |
| `/health` | GET | Health check |
| `/v1/chat` | POST | Unified chat (auto-routes: RAG / Live / Research) |
| `/v1/chat/stream` | POST | Streaming chat (SSE) |
| `/auth/register` | POST | Register new user |
| `/auth/login` | POST | Login |
| `/auth/me` | GET | Current user profile |
| `/chat/history` | GET | List conversations |
| `/chat/{id}` | GET | Get conversation with messages |
| `/chat/{id}/rename` | PUT | Rename conversation |
| `/feedback` | POST | Rate a message |
| `/memories` | GET/POST | Long-term memory management |
| `/ingest` | POST | Upload document (.txt, .md, .pdf) |
| `/admin/users` | GET | List users (admin) |
| `/admin/stats` | GET | System stats (admin) |
| `/kb` | GET/POST | Knowledge base management |
| `/metrics` | GET | Metrics endpoint |

### Example: Chat

```bash
curl -X POST http://localhost:8002/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is self-attention?", "mode": "auto"}'
```

Response:

```json
{
  "answer": "Self-attention computes weighted sums of value vectors based on query-key similarity scores...",
  "sources": [
    {
      "content": "The attention mechanism computes weighted sums based on query key similarity...",
      "relevance": 0.92,
      "source": "chroma_db"
    }
  ],
  "route": "rag",
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_used": "GroqLLM"
}
```

### Example: Streaming Chat

```bash
curl -X POST http://localhost:8002/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "Explain quantum computing", "mode": "auto"}'
```

SSE events: `route` → `sources` → `token` (one per word) → `done`

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch 2.x, NumPy |
| **Transformer** | Custom encoder-decoder (no HuggingFace) |
| **Vector Stores** | ChromaDB, Qdrant, BM25 |
| **LLM Backends** | Groq API (llama-3.1-8b), Ollama (phi3, llama3.2, gemma, mistral) |
| **API Framework** | FastAPI, Uvicorn, SSE |
| **Database** | SQLAlchemy 2.x, Alembic, SQLite (dev), PostgreSQL/asyncpg (prod) |
| **Cache** | Redis 5.x (with hiredis) |
| **Auth** | HMAC-SHA256 JWT, bcrypt |
| **Search** | DuckDuckGo, Wikipedia API, BeautifulSoup4 |
| **Object Storage** | MinIO (S3-compatible) |
| **Task Queue** | Celery (Redis broker) |
| **Monitoring** | OpenTelemetry, custom MetricsCollector |
| **Testing** | pytest, pytest-asyncio, pytest-cov, httpx |
| **Container** | Docker, Docker Compose |
| **Frontend** | Vanilla HTML/CSS/JS, marked.js, highlight.js |
| **Validation** | Pydantic v2, Pydantic-settings |

---

## Database

7 tables managed via SQLAlchemy 2.x + Alembic migrations:

| Table | Purpose |
|-------|---------|
| `users` | Authentication & profiles |
| `user_settings` | Per-user preferences (theme, default mode) |
| `conversations` | Chat sessions |
| `messages` | Messages with sources JSON, metadata |
| `feedback` | User ratings & comments |
| `user_memory` | Long-term per-user key-value memory |
| `knowledge_bases` | Multi-KB with metadata |
| `kb_user_permissions` | KB access control (read/write/admin) |

**Migrations:** `alembic/versions/` — 3 versions covering initial schema, user memory, and knowledge bases.

---

## Configuration

All configuration via `.env` (see `.env.example` for defaults):

| Key | Default | Description |
|-----|---------|-------------|
| `DATABASE_URL` | `sqlite:///rag.db` | Database connection string |
| `GROQ_API_KEY` | `""` | Groq API key (disables Ollama when set) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server endpoint |
| `PREFERRED_MODELS` | `phi3:mini,phi4,...` | Ordered LLM model priority list |
| `MAX_MODEL_SIZE_GB` | `5.0` | Skip models larger than this |
| `JWT_SECRET` | `change-me...` | HMAC signing key |
| `JWT_EXPIRY_HOURS` | `24` | Token lifetime |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector store endpoint |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persist directory |
| `CHUNK_SIZE` | `512` | Document chunk size (words) |
| `CHUNK_OVERLAP` | `64` | Chunk overlap (words) |
| `S3_ENDPOINT` | `http://localhost:9000` | MinIO S3 endpoint |
| `OTLP_ENDPOINT` | `""` | OpenTelemetry collector |
| `SENTRY_DSN` | `""` | Sentry DSN |

---

## Recent Fixes

| Issue | Fix |
|-------|-----|
| `/v1/chat` returning "Is there something I can help you with?" | Replaced `select_prompt()` with `_NO_CONTEXT_PROMPT_TEMPLATE` when no KB sources exist — no more "use the context" instructions without context |
| 422 validation errors on `message` field | Both `message` and `question` default to `""` in Pydantic model |
| MockLLM silently serving static text | `MockLLM.invoke()` now raises `RuntimeError` |
| Frontend showing placeholder before response | Bot message created only on first real SSE event |
| Greeting misclassification | `is_fast_path()` restricted to actual greetings only |

---

## License

MIT
