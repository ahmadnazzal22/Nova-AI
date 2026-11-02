# Custom Transformer RAG

<p align="center">
  <strong>A Custom PyTorch Transformer Connected to an Agentic Retrieval-Augmented Generation System</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=flat&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/RAG-Custom%20Architecture-7c3aed?style=flat" alt="RAG"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="MIT License"/>
</p>

---

## Overview

**Custom Transformer RAG** is a Python project that combines a Transformer architecture implemented with PyTorch and a Retrieval-Augmented Generation (RAG) system.

The project explores how custom Transformer components can be connected to a document retrieval and question-answering pipeline, with support for multiple query routes, language model integrations, and agent-based research workflows.

The system is organized into two main parts:

1. **Part 1 — Custom Transformer:** Transformer components implemented using PyTorch.
2. **Part 2 — Agentic RAG:** Retrieval, orchestration, API, and research functionality.

---

## Key Features

### 1. Custom Transformer

- Multi-head attention.
- Positional encoding.
- Encoder-decoder architecture.
- Tokenization and embedding components.
- Feed-forward layers.
- Training and checkpointing utilities.
- Embedding extraction for downstream retrieval.

### 2. Multi-Route Query Engine

The RAG system is designed to route questions into different processing modes:

| Mode | Purpose |
|---|---|
| RAG | Retrieve relevant information from a knowledge base. |
| Live | Search external web sources. |
| Research | Break complex questions into smaller research tasks and combine the results. |

### 3. Hybrid Retrieval

- Dense vector retrieval.
- BM25 keyword search.
- Hybrid retrieval combining multiple search methods.
- Result ranking and reranking.
- ChromaDB and Qdrant integrations.

### 4. Language Model Integration

- Groq API integration.
- Ollama integration for local model workflows.
- Model selection and fallback handling.
- Circuit-breaker logic.
- Streaming response support.

### 5. Research & Web Search

- Multi-step research workflow.
- Query decomposition.
- DuckDuckGo and Wikipedia search integrations.
- Source collection and response synthesis.
- Domain filtering support.

### 6. API & Streaming

- FastAPI-based API services.
- Unified chat endpoint.
- Server-Sent Events (SSE) streaming.
- Document ingestion endpoints.
- Authentication and conversation endpoints.

### 7. Memory & Knowledge Bases

- Conversation history.
- Long-term memory functionality.
- Multiple knowledge bases.
- Per-user knowledge-base permissions.
- Document chunking and ingestion.

### 8. Monitoring & Infrastructure

- Logging and query metrics.
- OpenTelemetry integration.
- Redis caching.
- Celery background tasks.
- Docker Compose configuration.
- PostgreSQL and Qdrant integrations.

---

## Architecture

The project is divided into two main components.

```text
                 USER QUERY
                     |
                     v
       +---------------------------+
       |    Custom Transformer     |
       |                           |
       | Tokenizer                 |
       | Embeddings                |
       | Attention                 |
       | Encoder / Decoder         |
       +-------------+-------------+
                     |
                     v
       +---------------------------+
       |       RAG System          |
       |                           |
       | API Gateway               |
       | Query Intelligence        |
       | Orchestrator              |
       +-------------+-------------+
                     |
          +----------+----------+
          |          |          |
          v          v          v
       +------+  +------+  +----------+
       | RAG  |  | Live |  | Research |
       +------+  +------+  +----------+
          |          |          |
          +----------+----------+
                     |
                     v
             +---------------+
             | LLM Service   |
             | Groq / Ollama |
             +-------+-------+
                     |
                     v
             +---------------+
             | Response      |
             | Formatter     |
             +-------+-------+
                     |
                     v
                  ANSWER
```

### Query Processing Flow

1. Receive the user's question.
2. Detect the query type and intent.
3. Select the appropriate processing route.
4. Retrieve relevant information or perform web research.
5. Generate a response using the configured language model.
6. Format and return the response.

---

## Project Structure

```text
custom-transformer-rag/
│
├── part1_transformer/
│   ├── attention.py
│   ├── transformer.py
│   ├── tokenizer.py
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── feed_forward.py
│   ├── config.py
│   ├── dataset.py
│   ├── train.py
│   ├── inference_engine.py
│   ├── exceptions.py
│   └── logger.py
│
├── part2_rag/
│   ├── api_gateway/
│   ├── orchestrator/
│   ├── llm/
│   ├── retrieval/
│   ├── ranking/
│   ├── auth/
│   ├── schemas/
│   ├── cache/
│   ├── context/
│   ├── citation/
│   ├── monitoring/
│   ├── storage/
│   ├── worker/
│   ├── rag_agent.py
│   ├── research_agent.py
│   ├── web_search.py
│   ├── web_loader.py
│   ├── chunker.py
│   ├── custom_embeddings.py
│   └── database.py
│
├── tests/
├── static/
├── alembic/
├── scripts/
├── data/
├── samples/
│
├── run_gateway.py
├── run_api.py
├── run_legacy.py
├── run_celery_worker.py
│
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── pytest.ini
└── README.md
```

*The structure above is a simplified overview based on the supplied project description. Check the actual repository before treating every listed path as present.*

---

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x, NumPy |
| Transformer | Custom Transformer architecture |
| API Framework | FastAPI, Uvicorn |
| Vector Databases | ChromaDB, Qdrant |
| Retrieval | BM25, Hybrid Retrieval |
| LLM Providers | Groq API, Ollama |
| Database | SQLAlchemy, SQLite, PostgreSQL |
| Migrations | Alembic |
| Caching | Redis |
| Authentication | JWT, bcrypt |
| Web Search | DuckDuckGo, Wikipedia |
| Background Tasks | Celery |
| Monitoring | OpenTelemetry |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Containers | Docker, Docker Compose |
| Frontend | HTML, CSS, JavaScript |

---

## Quick Start

### Prerequisites

- Python 3.10 or later.
- pip.
- PyTorch.
- Optional: Groq API key.
- Optional: Ollama for local language models.
- Optional: Redis, Qdrant, PostgreSQL, and Docker for additional services.

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd custom-transformer-rag
```

Replace `<YOUR_REPOSITORY_URL>` with the actual repository URL.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example configuration:

**Windows:**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

Configure the required API keys and service URLs in `.env`.

Never commit real secrets or credentials.

---

## Running the Project

### Train the Custom Transformer

```bash
python -m part1_transformer.train
```

The training workflow is designed to save model checkpoints and embedding outputs, depending on the configured training process.

### Start the API Gateway

```bash
python run_gateway.py
```

The supplied project configuration identifies port `8002` for the gateway.

Open the API documentation at:

```text
http://localhost:8002/docs
```

### Start the Legacy API

```bash
python run_legacy.py
```

The supplied project description identifies port `8000` for the legacy API.

### Run with Docker Compose

```bash
docker-compose up --build
```

The provided configuration is intended to support the project's containerized services.

---

## API Reference

The project description includes these API operations:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Frontend interface |
| `/health` | GET | Health check |
| `/v1/chat` | POST | Unified chat interface |
| `/v1/chat/stream` | POST | Streaming chat |
| `/auth/register` | POST | User registration |
| `/auth/login` | POST | User login |
| `/auth/me` | GET | Current user |
| `/chat/history` | GET | Conversation history |
| `/chat/{id}` | GET | Conversation details |
| `/feedback` | POST | Message feedback |
| `/memories` | GET / POST | Memory operations |
| `/ingest` | POST | Document ingestion |
| `/kb` | GET / POST | Knowledge-base management |
| `/metrics` | GET | Monitoring metrics |

Check the running API documentation for the actual available routes and request schemas.

---

## Example: Chat Request

```bash
curl -X POST http://localhost:8002/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is self-attention?", "mode": "auto"}'
```

The response format depends on the configured API implementation and selected processing route.

---

## Configuration

The project description lists the following environment variables:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Database connection |
| `GROQ_API_KEY` | Groq API authentication |
| `OLLAMA_BASE_URL` | Local Ollama endpoint |
| `PREFERRED_MODELS` | Preferred model order |
| `MAX_MODEL_SIZE_GB` | Model size limit |
| `JWT_SECRET` | JWT signing secret |
| `JWT_EXPIRY_HOURS` | Token lifetime |
| `QDRANT_URL` | Qdrant endpoint |
| `REDIS_URL` | Redis connection |
| `CHROMA_DB_PATH` | ChromaDB storage path |
| `CHUNK_SIZE` | Document chunk size |
| `CHUNK_OVERLAP` | Chunk overlap |
| `S3_ENDPOINT` | Object storage endpoint |
| `OTLP_ENDPOINT` | OpenTelemetry endpoint |
| `SENTRY_DSN` | Error monitoring configuration |

See `.env.example` for the actual configuration supported by the repository.

---

## Testing

If the test suite and dependencies are installed, run:

```bash
pytest tests/ -v
```

For coverage:

```bash
pytest tests/ --cov=part1_transformer --cov=part2_rag
```

Test counts and coverage should be reported only after running the tests.

---

## Development Focus

Potential areas for further development include:

- Improving Transformer training and evaluation.
- Measuring embedding quality.
- Improving retrieval relevance and reranking.
- Expanding knowledge-base management.
- Enhancing research workflows.
- Improving API reliability and test coverage.
- Optimizing inference and resource usage.

---

## License

MIT License.

---

<p align="center">
  <strong>Custom Transformer RAG</strong><br/>
  <em>Building the connection between custom deep learning architectures and intelligent information retrieval.</em>
</p>
