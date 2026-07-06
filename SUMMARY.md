# 18 — RAG System Summary

## Current State (June 25)

### Done
- **ChromaDB**: 44,170 unique chunks ingested from 75 Wikipedia articles (4.4x target)
- **Tests**: 83/83 pass (all tests: API, attention, chunker, embeddings, inference engine, RAG agent, tokenizer, training, transformer, KV cache)
- **Batched embedding**: `embed_documents()` processes in batches of 64 (fixes OOM on 37k docs)
- **Batched ingestion**: Pipeline ingests 500 docs/batch to avoid ChromaDB overload
- **FastAPI endpoints**: `/query`, `/ingest`, `/query/stream`, `/chat`, `/chat/stream`, `/health` — all with Pydantic v2, SSE streaming, request UUID tracing
- **MockLLM**: Clean summarized output as temporary fallback until small Ollama model is pulled
- **`_clean_text()`**: Collapses noise/deduplication in retrieved context
- **Singleton pattern**: `RAGAgent` and `TransformerEmbeddings` prevent duplicate load
- **`scripts/build_dataset.py`**: Fetches Wikipedia articles, cleans wikitext, chunks (300/60), augments x6, deduplicates by MD5, ingests

### Blocked
- Ollama available (`gemma4:latest` 9.6GB) exceeds 5GB limit; MockLLM fallback active
- Transformer checkpoint: vocab=5004, d_model=256, trained on limited data — sufficient for embedding, not generation
- Many ingested articles are Wikipedia "Category:" pages (poor answer quality)

### Next Steps
1. **Pull** `ollama pull llama3.2:1b` — auto-replaces MockLLM with real LLM
2. **Run API** — `uvicorn part2_rag.api:app --reload` port 8000
3. **Open frontend** — `http://localhost:8002` (served by FastAPI at `/`)
4. **Re-fetch articles** — Filter out Wikipedia Category: pages for better quality data
5. **Train transformer** — `python -m part1_transformer.train` on the new 44k chunk dataset
6. **Docker** — `docker-compose build && docker-compose up`

### Files
- `part2_rag/rag_agent.py` — RAGAgent, MockLLM, OllamaLLM, `_clean_text()`, dedup, PROMPT_TEMPLATE
- `part2_rag/custom_embeddings.py` — Singleton TransformerEmbeddings, batch=64, mean pooling, query cache
- `part2_rag/api.py` — FastAPI app + streaming SSE
- `static/index.html` — Frontend SPA
- `scripts/build_dataset.py` — Full data pipeline
- `data/processed/chunks.txt` — 60,620 raw chunks (44,170 unique after MD5 dedup)
