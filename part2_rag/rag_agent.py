import json
import os
import re
import time
import uuid
import threading
import concurrent.futures
from typing import Any, Generator

import httpx
from langchain_chroma import Chroma
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .custom_embeddings import TransformerEmbeddings
from .chunker import ingest_document, SUPPORTED_EXTS
from .web_loader import WebLoader
from .web_search import WebSearch
from .reranker import Reranker, enrich_sources_with_scoring
from .tool_decision_llm import ToolDecisionLLM, ToolDecision
from .fast_path import is_fast_path, GREETINGS as FAST_PATH_GREETINGS
from .memory_manager import MemoryManager
from .response_validator import ResponseValidator
from .response_formatter import format_response
from .prompt_router import select_prompt
from .exceptions import VectorstoreError, RetrievalError, LLMError
from .logger import get_logger, set_correlation_id, timed, log_pipeline
from .query_intelligence import QueryIntelligence
from .retrieval.hybrid_retriever import HybridRetriever
from .context_compressor import ContextCompressor

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are a professional AI assistant. Answer using both your knowledge and the provided context when relevant.
Cite sources with [1], [2], etc.

MANDATORY FORMATTING RULES:
- Be concise. Maximum 3-4 paragraphs. No repetition. No redundant headers.
- NEVER repeat the user's question at the start of your answer.
- Start with a short introduction (1-2 sentences).
- Use ## for main headings and ### for subheadings.
- Use paragraphs, bullet points (-), and numbered lists (1.).
- Use the same language as the user.
- Use Markdown formatting ONLY — no decorative separators like ===== or -----.
- Keep paragraphs short (2-4 sentences).
- If information is insufficient, state it clearly — do NOT fabricate.
- Integrate context naturally — do NOT copy it verbatim.

Use this concise structure (pick 2-3 relevant sections):
## Introduction
## Key Points
## Conclusion

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

LIVE_PROMPT_TEMPLATE = """You are a professional AI assistant with access to live web data.
Answer using both your knowledge and the provided search results when relevant.
Cite sources with [1], [2], etc.

MANDATORY FORMATTING RULES:
- Be concise. Maximum 3-4 paragraphs. No repetition. No redundant headers.
- NEVER repeat the user's question at the start of your answer.
- Start with a short introduction (1-2 sentences).
- Use ## for main headings and ### for subheadings.
- Use paragraphs, bullet points (-), and numbered lists (1.).
- Use the same language as the user.
- Use Markdown formatting ONLY — no decorative separators.
- Keep paragraphs short (2-4 sentences).
- If information is insufficient, state it clearly — do NOT fabricate.
- Integrate context naturally — do NOT copy it verbatim.

Use this concise structure (pick 2-3 relevant sections):
## Introduction
## Key Points
## Conclusion

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

DIRECT_PROMPT_TEMPLATE = """You are a professional AI assistant.

MANDATORY FORMATTING RULES:
- Be concise. Maximum 3-4 paragraphs. No repetition. No redundant headers.
- NEVER repeat the user's question at the start of your answer.
- Start with a short introduction (1-2 sentences).
- Use ## for main headings and ### for subheadings.
- Use paragraphs, bullet points (-), and numbered lists (1.).
- Use the same language as the user.
- Use Markdown formatting ONLY — no decorative separators.
- Keep paragraphs short (2-4 sentences).
- If information is insufficient, state it clearly — do NOT fabricate.
- Be precise and factual.

Question:
{question}"""

RELATED_QUESTIONS_PROMPT = """Based on the following Q&A, suggest 3-5 related questions the user might ask next.
Return ONLY a JSON array of strings, one question per item.

Q: {question}
A: {answer}

Related questions (JSON array):"""

PREFERRED_MODELS = [
    "phi3:mini",
    "phi4",
    "llama3.2",
    "gemma4",
    "gemma3",
    "llama3.2:1b",
    "tinyllama",
    "mistral",
]
WEAK_MODELS = frozenset({"llama3.2:1b", "tinyllama"})
_MAX_MODEL_SIZE_GB = 5.0
_MAX_CONTEXT_CHARS = 6000
_QUERY_CACHE_TTL = 300
_LIVE_CHUNK_SIZE = 300
_LIVE_CHUNK_OVERLAP = 60

_CORE_CONCEPT_SNIPPETS: dict[str, str] = {
    "transformer": (
        "A Transformer is a neural network architecture that uses self-attention mechanisms "
        "to process sequential data in parallel. It was introduced in the paper "
        "'Attention Is All You Need' (Vaswani et al., 2017)."
    ),
    "self-attention": (
        "Self-attention computes a weighted sum over all positions in a sequence, where the "
        "weights are learned based on pairwise compatibility between positions."
    ),
    "attention": (
        "Attention is a mechanism that allows a model to focus on relevant parts of the input. "
        "Self-attention computes attention scores within a single sequence."
    ),
    "embedding": (
        "An embedding is a dense vector representation of discrete data in a continuous vector "
        "space, where semantic similarity corresponds to vector proximity."
    ),
    "tokenizer": (
        "A tokenizer converts raw text into a sequence of tokens (subwords, words, or characters) "
        "that can be processed by a language model, using a learned vocabulary."
    ),
    "rag": (
        "Retrieval-Augmented Generation (RAG) combines a retrieval system (e.g., vector search) "
        "with a generative language model to produce answers grounded in external knowledge."
    ),
}


_EXPLANATION_STARTS = (
    "what is", "what's", "what are", "what does",
    "explain", "how to", "how do", "how does",
    "define", "describe", "why", "what is the difference",
)

_REAL_TIME_WORDS = frozenset({
    "latest", "news", "today", "current", "weather",
    "temperature", "price", "stock", "bitcoin", "ethereum",
    "crypto", "forecast", "humidity",
})


def _is_explanation_task(query: str) -> bool:
    q = query.lower().strip()
    if not q.startswith(_EXPLANATION_STARTS):
        return False
    words = set(q.split())
    return not bool(_REAL_TIME_WORDS & words)


_SPORTS_KEYWORDS = frozenset({
    "مباراة", "مباريات", "نتيجة", "موعد المباراة", "كرة القدم",
    "match", "score", "game today", "fixture",
})


def simple_router(query: str) -> str:
    q = query.strip().lower()
    if q in ("hi", "hello", "hey", "مرحبا", "هلا"):
        return "GREETING"
    if q in ("help", "مساعدة") or q.strip() in ("i need help", "can you help", "مساعده"):
        return "HELP"
    if any(kw in q for kw in _SPORTS_KEYWORDS):
        return "SPORTS"
    return "LLM"


def _inject_concept_snippets(question: str, prompt: str) -> str:
    lower_q = question.lower()
    snippets = []
    for keyword, snippet in _CORE_CONCEPT_SNIPPETS.items():
        if keyword in lower_q:
            snippets.append(snippet)
    if snippets:
        reference = "\n\n".join(snippets)
        prompt = f"Reference context:\n{reference}\n\n{prompt}"
    return prompt


def _clean_text(text: str) -> str:
    text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = text.strip()
    return text


def _deduplicate_docs(docs: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for d in docs:
        key = d.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(d)
    return result


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title", "")
        url = s.get("url", "")
        snippet = s.get("snippet", "")
        content = s.get("content", s.get("text", ""))
        display = content or snippet
        if display:
            display = display[:300]
        parts = [f"[{i}]"]
        if title:
            parts.append(title)
        if display:
            parts.append(display)
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def _format_context_with_citations(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", chunk.get("content", chunk.get("snippet", "")))
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def _parse_cited_answer(answer: str) -> str:
    answer = re.sub(r"\s+", " ", answer).strip()
    return answer


def _clean_context(chunks: list[dict]) -> list[dict]:
    seen: set[int] = set()
    result = []
    for c in chunks:
        text = c.get("text", c.get("content", c.get("snippet", "")))
        h = hash(text[:200].strip().lower())
        if h in seen:
            continue
        seen.add(h)
        cleaned = c.get("_clean_text", text)
        cleaned = re.sub(r"\[(?!\d+\])[^\]]*\]", "", cleaned)
        cleaned = re.sub(r"(?i)(click here|subscribe|newsletter|advertisement|sponsored|cookie policy|privacy policy)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        c["text"] = cleaned
        result.append(c)
    return result


def _is_sufficient_context(reranked_with_scores: list[tuple[dict, float]]) -> bool:
    if not reranked_with_scores:
        return False
    scores = [s for _, s in reranked_with_scores]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    # Dynamic threshold: at least one doc above 0.15 AND top score is within 50% of the max possible
    top_pct = max_score / max(scores) if scores else 0
    return max_score >= 0.15 and avg_score >= 0.08 and top_pct >= 0.3


class MockLLM(LLM):
    model: str = "mock"

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, **kwargs) -> str:
        if "CONTEXT:" in prompt and "No context available." not in prompt:
            ctx_match = re.search(r"CONTEXT:\s*(.*?)(?:\n\nQUESTION:|$)", prompt, re.DOTALL)
            if ctx_match:
                context = ctx_match.group(1).strip()
                summary = _clean_text(context)[:300]
                return f"Based on the provided information: {summary}"
        return "I don't have enough information to answer that question."

    def _stream(self, prompt: str, **kwargs) -> Generator[GenerationChunk, None, None]:
        yield GenerationChunk(text=self._call(prompt, **kwargs))

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}


class GroqLLM(LLM):
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_tokens: int = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from groq import Groq
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    @property
    def _llm_type(self) -> str:
        return "groq"

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        """Split a combined prompt into (system_instructions, user_question).

        The templates have instructions followed by 'Question:' / 'QUESTION:' / 'ANSWER:'
        at the end. Everything before the last question marker is the system prompt.
        """
        # Try the last occurrence of common question delimiters
        for marker in ("QUESTION:", "Question:"):
            idx = prompt.rfind(marker)
            if idx != -1:
                system_part = prompt[:idx].strip()
                user_part = prompt[idx:].strip()
                return system_part, user_part
        # Fallback: entire prompt is user, empty system
        return "", prompt

    def _call(self, prompt: str, **kwargs) -> str:
        system, user = self._split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        logger.debug("GROQ SYSTEM PROMPT:\n%s", system[:1000])

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    def _stream(self, prompt: str, **kwargs) -> Generator[GenerationChunk, None, None]:
        system, user = self._split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        logger.debug("GROQ SYSTEM PROMPT:\n%s", system[:1000])

        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                yield GenerationChunk(text=delta)

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}


class RAGAgent:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, embeddings: TransformerEmbeddings | None = None, persist_dir: str | None = None, config: Any = None, llm=None):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._chroma_lock = threading.Lock()

        self.config = config
        self.embeddings = embeddings or TransformerEmbeddings()
        if config:
            self.persist_dir = persist_dir or config.chroma_db_path
            self.collection_name = config.chroma_collection
            self.retrieval_k = config.retrieval_k
        else:
            self.persist_dir = persist_dir or os.getenv("CHROMA_DB_PATH", "./chroma_db")
            self.collection_name = os.getenv("CHROMA_COLLECTION", "langchain")
            self.retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))

        self.db = self._load_vectorstore()
        self.llm = llm
        self.web_loader = WebLoader(max_results=5)
        self.web_search = WebSearch(max_results=5)
        self.reranker = Reranker(embeddings_model=self.embeddings, top_k=self.retrieval_k)
        self.tool_decider = ToolDecisionLLM(llm=llm)
        self._memory_manager: MemoryManager | None = None
        self._query_cache: dict[str, dict] = {}

        self.query_intelligence = QueryIntelligence(llm=llm, embeddings=self.embeddings)
        self.hybrid_retriever = HybridRetriever(
            vector_store=self.db,
            embedding_fn=self.embeddings.embed_query,
        )
        self.context_compressor = ContextCompressor(
            embeddings=self.embeddings,
            llm=llm,
        )
        self._rebuild_hybrid_index()

    def _cache_get(self, question: str) -> dict | None:
        entry = self._query_cache.get(question)
        if entry:
            ts = entry.get("_ts", 0)
            if time.time() - ts < _QUERY_CACHE_TTL:
                result = {k: v for k, v in entry.items() if k != "_ts"}
                result["cached"] = True
                return result
            del self._query_cache[question]
        return None

    def _cache_set(self, question: str, data: dict):
        data["_ts"] = time.time()
        self._query_cache[question] = data
        if len(self._query_cache) > 100:
            stale = [k for k, v in self._query_cache.items() if time.time() - v.get("_ts", 0) > _QUERY_CACHE_TTL]
            for k in stale:
                del self._query_cache[k]

    def _rebuild_hybrid_index(self):
        try:
            all_docs = self.db._collection.get()
            texts = all_docs.get("documents", [])
            metadatas = all_docs.get("metadatas", [])
            if texts:
                docs = []
                for i, t in enumerate(texts):
                    meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                    docs.append({"text": t, **meta})
                self.hybrid_retriever.rebuild_index(docs)
                logger.info("Hybrid retriever index rebuilt: %d docs", len(docs))
        except Exception as e:
            logger.warning("Failed to rebuild hybrid index: %s", e)

    def _load_vectorstore(self) -> Chroma:
        try:
            db = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            count = db._collection.count()
            logger.info("Vectorstore loaded: %d docs", count)
            return db
        except Exception as e:
            raise VectorstoreError("Failed to load vectorstore") from e

    def add_documents(self, texts: list[str]) -> int:
        with self._chroma_lock:
            try:
                cleaned = [_clean_text(t) for t in texts]
                cleaned = [t for t in cleaned if len(t) > 10]
                if not cleaned:
                    return self.db._collection.count()
                self.db.add_texts(texts=cleaned)
                count = self.db._collection.count()
                logger.info("Added %d documents | total: %d", len(cleaned), count)
                return count
            except Exception as e:
                raise VectorstoreError("Failed to add documents") from e
            finally:
                try:
                    self._rebuild_hybrid_index()
                except Exception:
                    pass

    def ingest_file(self, filepath: str, chunk_size: int = 512, chunk_overlap: int = 64) -> dict:
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in SUPPORTED_EXTS:
            raise VectorstoreError(f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTS)}")

        chunks = ingest_document(filepath, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            with self._chroma_lock:
                total = self.db._collection.count()
            return {"filename": os.path.basename(filepath), "chunks": 0, "total_docs": total}

        total = self.add_documents(chunks)
        return {
            "filename": os.path.basename(filepath),
            "chunks": len(chunks),
            "total_docs": total,
        }

    def _get_llm(self) -> LLM:
        groq_key = os.environ.get("GROQ_API_KEY") or (self.config and getattr(self.config, 'groq_api_key', None))
        if groq_key:
            logger.info("Using Groq API with model: llama-3.1-8b-instant (tier=strong)")
            return GroqLLM(model="llama-3.1-8b-instant", temperature=0.3)

        if self.config:
            ollama_url = self.config.ollama_base_url
            ollama_timeout = self.config.ollama_timeout
        else:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "60"))

        try:
            r = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
        except httpx.ConnectError:
            logger.warning("Ollama not running; using MockLLM fallback")
            return MockLLM()
        except Exception as e:
            logger.warning("Ollama unavailable (%s); using MockLLM fallback", str(e)[:40])
            return MockLLM()

        max_size = _MAX_MODEL_SIZE_GB * 1_000_000_000
        if self.config and hasattr(self.config, 'max_model_size_gb'):
            max_size = self.config.max_model_size_gb * 1_000_000_000

        models = r.json().get("models", [])
        small_models = [m for m in models if m.get("size", 0) < max_size]

        def normalize(name: str) -> str:
            return name.removesuffix(":latest")

        available = {normalize(m["name"]) for m in small_models}
        available_raw = {m["name"] for m in small_models}

        strong_available = {m for m in available if m not in WEAK_MODELS}
        preferred_strong = [m for m in PREFERRED_MODELS if m in (strong_available if strong_available else available)]

        model_name = None
        for pref in preferred_strong:
            if pref in available:
                model_name = pref
                break

        if model_name is None and small_models:
            smallest = min(small_models, key=lambda m: m.get("size", 0))
            model_name = smallest["name"]
            logger.info("Preferred models not found; using smallest available: %s", model_name)

        if model_name is None:
            logger.warning("No small Ollama models found (<%dGB); using MockLLM fallback", max_size / 1e9)
            return MockLLM()

        tier = "weak" if model_name in WEAK_MODELS else "strong"
        logger.info("Selected LLM model: %s  (tier=%s, url=%s)", model_name, tier, ollama_url)

        from langchain_ollama import OllamaLLM
        return OllamaLLM(model=model_name, temperature=0.3, timeout=ollama_timeout)

    def _ensure_llm(self) -> Any:
        if self.llm is None:
            self.llm = self._get_llm()
        if self.llm is not None:
            self.tool_decider.set_llm(self.llm)
            self.query_intelligence.set_llm(self.llm)
            self.context_compressor._llm = self.llm
        return self.llm

    def _get_memory_manager(self):
        from .database import get_session
        session = get_session()
        return MemoryManager(session), session

    def _inject_memories(self, question: str, prompt: str, user_id: int | None = None) -> str:
        if user_id is None:
            return prompt
        mm, session = self._get_memory_manager()
        try:
            context = mm.load_context(user_id, limit=5)
            if context:
                prompt = f"{context}\n\n{prompt}"
        finally:
            session.close()
        return prompt

    def _extract_memories(self, question: str, answer: str, user_id: int | None = None):
        if user_id is None:
            return
        mm, session = self._get_memory_manager()
        try:
            mm.extract_and_store(user_id, question, answer, self.llm)
        finally:
            session.close()

    @timed()
    def _direct_llm_answer(self, question: str, user_id: int | None = None) -> dict:
        is_greeting = question.lower().strip() in FAST_PATH_GREETINGS
        template = select_prompt(question)
        prompt = template.format(question=question)
        if not is_greeting:
            prompt = _inject_concept_snippets(question, prompt)
        prompt = self._inject_memories(question, prompt, user_id)
        answer = self.llm.invoke(prompt)
        logger.debug("RAW LLM ANSWER: %.200s", answer)
        answer = format_response(answer, question)
        logger.debug("FORMATTED ANSWER: %.200s", answer)
        if not is_greeting:
            invalid = not ResponseValidator.is_valid_answer(question, answer)
            low_conf = ResponseValidator.low_confidence(answer) and not _is_explanation_task(question)
            if invalid or low_conf:
                logger.info("Regenerating low-quality answer for: %.60s", question)
                fallback = ResponseValidator.build_fallback_prompt(question)
                answer = self.llm.invoke(fallback)
                if "i am not certain" in answer.lower() or "i am not sure" in answer.lower():
                    logger.info("Regenerated answer still uncertain; second attempt: %.60s", question)
                    answer = self.llm.invoke(
                        f"Question: {question}\n\n"
                        f"Answer correctly and confidently. Do NOT say you are not certain."
                    )
        logger.debug("RAW LLM ANSWER (post-regeneration): %.200s", answer)
        answer = format_response(answer, question)
        logger.debug("FORMATTED ANSWER: %.200s", answer)
        self._extract_memories(question, answer, user_id)
        return {"answer": answer, "sources": [], "cached": False, "tool_used": None}

    def _validate_and_regenerate(self, question: str, answer: str) -> str:
        is_greeting = question.lower().strip() in FAST_PATH_GREETINGS
        if not is_greeting:
            invalid = not ResponseValidator.is_valid_answer(question, answer)
            low_conf = ResponseValidator.low_confidence(answer) and not _is_explanation_task(question)
            if invalid or low_conf:
                logger.info("Regenerating low-quality answer for: %.60s", question)
                fallback = ResponseValidator.build_fallback_prompt(question)
                answer = self.llm.invoke(fallback)
                if "i am not certain" in answer.lower() or "i am not sure" in answer.lower():
                    logger.info("Regenerated answer still uncertain; second attempt: %.60s", question)
                    answer = self.llm.invoke(
                        f"Question: {question}\n\n"
                        f"Answer correctly and confidently. Do NOT say you are not certain."
                    )
        logger.debug("RAW LLM ANSWER (validate/regenerate): %.200s", answer)
        result = format_response(answer, question)
        logger.debug("FORMATTED ANSWER: %.200s", result)
        return result

    def _generate_related_questions(self, question: str, answer: str) -> list[str]:
        try:
            prompt = RELATED_QUESTIONS_PROMPT.format(question=question, answer=answer[:500])
            response = self.llm.invoke(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
            questions = json.loads(response)
            if isinstance(questions, list):
                return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:5]
        except Exception as e:
            logger.debug("Failed to generate related questions: %s", e)
        return []

    @timed()
    def _build_prompt(self, question: str) -> tuple[str, list[str]]:
        qi_result = self.query_intelligence.process(question)
        logger.debug(
            "QueryIntelligence: intent=%s keywords=%d multi_queries=%d",
            qi_result.intent, len(qi_result.keywords), len(qi_result.multi_queries),
        )

        extra_queries = qi_result.multi_queries[1:] if len(qi_result.multi_queries) > 1 else None
        rewritten = qi_result.rewritten or question

        with self._chroma_lock:
            results = self.hybrid_retriever.search(
                rewritten,
                k=self.retrieval_k + 5,
                extra_queries=extra_queries,
            )

        if not results:
            return "", []

        scored = self.reranker.rank_with_scores(question, results, k=self.retrieval_k + 2)

        if not scored:
            return "", []

        top_chunks = [doc for doc, _ in scored]

        compressed = self.context_compressor.compress(
            top_chunks, question, max_chars=_MAX_CONTEXT_CHARS
        )

        context = self.context_compressor.format_context(compressed)
        if len(context) > _MAX_CONTEXT_CHARS:
            context = context[:_MAX_CONTEXT_CHARS] + "..."

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        source_texts = []
        for c in compressed:
            t = c.get("text", c.get("content", c.get("snippet", "")))
            if t:
                source_texts.append(t)

        return prompt, source_texts[:self.retrieval_k]

    def _generate_cited_answer(self, prompt: str, context_chunks: list[dict]) -> tuple[str, list[dict]]:
        answer = self.llm.invoke(prompt)
        answer = _parse_cited_answer(answer)
        used_sources = []
        for i, chunk in enumerate(context_chunks, 1):
            ref = f"[{i}]"
            if ref in answer:
                used_sources.append(chunk)
        if not used_sources and context_chunks:
            used_sources = context_chunks[:self.retrieval_k]
        return answer, used_sources

    def _stream_cited_answer(self, prompt: str) -> Generator[str, None, None]:
        for token in self.llm.stream(prompt):
            yield token

    @timed("INFO")
    def query(self, question: str, user_id: int | None = None) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("start", {"correlation_id": cid, "question": question[:60]}, logger)

        cached = self._cache_get(question)
        if cached:
            log_pipeline("cache_hit", {"question": question[:60]}, logger)
            return cached

        route = simple_router(question)
        if route == "GREETING":
            return {"answer": "Hello! How can I help you today?", "sources": [], "cached": False, "tool_used": None}
        if route == "HELP":
            return {"answer": "Sure! Tell me what you need and I will help you.", "sources": [], "cached": False, "tool_used": None}
        if route == "SPORTS":
            return {"answer": "عذراً، أنا متخصص بالذكاء الاصطناعي فقط ولا أملك بيانات رياضية. ابحث في Google أو تطبيق ESPN.", "sources": [], "cached": False, "tool_used": None}

        try:
            self._ensure_llm()

            qi_result = self.query_intelligence.process(question)
            log_pipeline("query_intelligence", {
                "intent": qi_result.intent,
                "keywords": qi_result.keywords[:5],
                "multi": len(qi_result.multi_queries),
            }, logger)

            decision = self.tool_decider.decide(question)
            log_pipeline("tool_decision", {"tool": decision.tool, "use_tools": decision.use_tools}, logger)

            search_q = decision.rewritten_query or qi_result.rewritten or question
            log_pipeline("kb_retrieval", {"query": search_q[:60]}, logger)
            prompt, sources = self._build_prompt(search_q)

            if not sources:
                result = {"answer": "", "sources": [], "cached": False, "tool_used": "kb", "no_results": True}
            else:
                prompt = self._inject_memories(question, prompt, user_id)
                answer = self.llm.invoke(prompt)
                answer = self._validate_and_regenerate(question, answer)
                self._extract_memories(question, answer, user_id)
                result = {"answer": answer, "sources": sources, "cached": False, "tool_used": "kb"}

            related = self._generate_related_questions(question, result.get("answer", ""))
            if related:
                result["related_questions"] = related
            self._cache_set(question, result)
            log_pipeline("complete", {"correlation_id": cid}, logger)
            return result
        except LLMError:
            raise
        except Exception as e:
            raise RetrievalError(f"Query failed: {question[:50]}") from e

    @timed("INFO")
    def query_live(self, question: str, top_k: int | None = None, user_id: int | None = None) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("live_start", {"correlation_id": cid, "question": question[:60]}, logger)

        cached = self._cache_get(question)
        if cached:
            log_pipeline("live_cache_hit", {"question": question[:60]}, logger)
            return cached

        route = simple_router(question)
        if route == "GREETING":
            return {"answer": "Hello! How can I help you today?", "sources": [], "live": True, "cached": False, "tool_used": None}
        if route == "HELP":
            return {"answer": "Sure! Tell me what you need and I will help you.", "sources": [], "live": True, "cached": False, "tool_used": None}
        if route == "SPORTS":
            return {"answer": "عذراً، أنا متخصص بالذكاء الاصطناعي فقط ولا أملك بيانات رياضية. ابحث في Google أو تطبيق ESPN.", "sources": [], "live": True, "cached": False, "tool_used": None}

        try:
            self._ensure_llm()

            qi_result = self.query_intelligence.process(question)
            log_pipeline("query_intelligence", {
                "intent": qi_result.intent,
                "keywords": qi_result.keywords[:5],
                "multi": len(qi_result.multi_queries),
            }, logger)

            if is_fast_path(question):
                log_pipeline("live_fast_path", {"question": question[:60]}, logger)
                result = self._direct_llm_answer(question, user_id)
                result["live"] = True
                related = self._generate_related_questions(question, result["answer"])
                if related:
                    result["related_questions"] = related
                self._cache_set(question, result)
                return result

            decision = self.tool_decider.decide(question)

            if not decision.use_tools or decision.tool not in ("web_search", "file_search"):
                logger.info("Live direct LLM answer: %.60s", question)
                result = self._direct_llm_answer(question, user_id)
                result["live"] = True
                return result

            k = top_k or self.retrieval_k
            search_q = decision.rewritten_query or qi_result.rewritten or question

            search_results = self.web_search.search_and_fetch(
                search_q, max_results=min(k + 2, 5), max_chars_per_page=2000,
            )

            if not search_results:
                logger.info("No live data, using LLM knowledge for: %.60s", question)
                result = self._direct_llm_answer(question, user_id)
                result["live"] = True
                return result

            chunks = []
            for sr in search_results:
                text = sr.content or sr.snippet
                if text:
                    chunks.append({"text": text, "title": sr.title, "url": sr.url, "snippet": sr.snippet, "source": sr.source})

            scored = self.reranker.rank_with_scores(question, chunks, k=k + 2)

            if not scored or max(s for _, s in scored) < 0.1:
                logger.info("Low-quality web results, using LLM knowledge for: %.60s", question)
                result = self._direct_llm_answer(question, user_id)
                result["live"] = True
                return result

            reranked = [doc for doc, _ in scored[:k]]
            reranked = _clean_context(reranked)

            compressed = self.context_compressor.compress(reranked, question)

            context = _format_context_with_citations(compressed)
            if len(context) > _MAX_CONTEXT_CHARS:
                context = context[:_MAX_CONTEXT_CHARS] + "..."

            prompt = LIVE_PROMPT_TEMPLATE.format(context=context, question=question)
            prompt = self._inject_memories(question, prompt, user_id)
            answer, cited_sources = self._generate_cited_answer(prompt, compressed)
            answer = self._validate_and_regenerate(question, answer)
            self._extract_memories(question, answer, user_id)

            source_list = []
            for s in enrich_sources_with_scoring(cited_sources):
                source_list.append({
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", s.get("text", ""))[:200],
                    "source": s.get("source", "web"),
                    "relevance_score": s.get("relevance_score", round(s.get("_relevance_score", 0), 4)),
                    "confidence_score": s.get("confidence_score", round(s.get("_confidence_score", 0), 4)),
                    "source_type": s.get("source_type", s.get("source", "web")),
                    "highlight_keywords": s.get("highlight_keywords", []),
                })

            result = {"answer": answer, "sources": source_list, "live": True, "cached": False, "tool_used": "web"}

            related = self._generate_related_questions(question, answer)
            if related:
                result["related_questions"] = related

            self._cache_set(question, result)
            log_pipeline("live_complete", {"correlation_id": cid}, logger)
            return result
        except LLMError:
            raise
        except Exception as e:
            raise RetrievalError(f"Live query failed: {question[:50]}") from e

    @timed("INFO")
    def query_live_stream(self, question: str, top_k: int | None = None, user_id: int | None = None) -> Generator[str, None, None]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("stream_start", {"correlation_id": cid, "question": question[:60]}, logger)

        route = simple_router(question)
        if route == "SPORTS":
            yield json.dumps({"type": "status", "message": "عذراً، أنا متخصص بالذكاء الاصطناعي فقط ولا أملك بيانات رياضية."}) + "\n"
            yield json.dumps({"type": "done", "data": "عذراً، أنا متخصص بالذكاء الاصطناعي فقط ولا أملك بيانات رياضية. ابحث في Google أو تطبيق ESPN."}) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            return

        try:
            self._ensure_llm()

            qi_result = self.query_intelligence.process(question)
            log_pipeline("stream_query_intelligence", {
                "intent": qi_result.intent,
                "keywords": qi_result.keywords[:5],
                "multi": len(qi_result.multi_queries),
            }, logger)

            if is_fast_path(question):
                yield json.dumps({"type": "status", "message": "Thinking..."}) + "\n"
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                answer_parts = []
                yield json.dumps({"type": "start"}) + "\n"
                for token in self.llm.stream(prompt):
                    answer_parts.append(token)
                    yield json.dumps({"type": "token", "data": token}) + "\n"
                answer = _parse_cited_answer("".join(answer_parts))
                answer = format_response(answer, question)
                self._extract_memories(question, answer, user_id)
                yield json.dumps({"type": "done", "data": answer}) + "\n"
                yield json.dumps({"type": "sources", "data": []}) + "\n"
                return

            decision = self.tool_decider.decide(question)
            log_pipeline("stream_tool_decision", {"tool": decision.tool}, logger)

            if not decision.use_tools or decision.tool not in ("web_search", "file_search"):
                yield json.dumps({"type": "status", "message": "Thinking..."}) + "\n"
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                answer_parts = []
                yield json.dumps({"type": "start"}) + "\n"
                for token in self.llm.stream(prompt):
                    answer_parts.append(token)
                    yield json.dumps({"type": "token", "data": token}) + "\n"
                answer = _parse_cited_answer("".join(answer_parts))
                answer = format_response(answer, question)
                self._extract_memories(question, answer, user_id)
                yield json.dumps({"type": "done", "data": answer}) + "\n"
                yield json.dumps({"type": "sources", "data": []}) + "\n"
                return

            k = top_k or self.retrieval_k
            search_q = qi_result.rewritten or question

            yield json.dumps({"type": "status", "message": "Searching the web..."}) + "\n"

            search_results = self.web_search.search_and_fetch(
                search_q, max_results=min(k + 2, 5), max_chars_per_page=2000,
            )

            if not search_results:
                yield json.dumps({"type": "status", "message": "Answering from my knowledge..."}) + "\n"
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                answer_parts = []
                yield json.dumps({"type": "start"}) + "\n"
                for token in self.llm.stream(prompt):
                    answer_parts.append(token)
                    yield json.dumps({"type": "token", "data": token}) + "\n"
                answer = _parse_cited_answer("".join(answer_parts))
                answer = format_response(answer, question)
                self._extract_memories(question, answer, user_id)
                yield json.dumps({"type": "done", "data": answer}) + "\n"
                yield json.dumps({"type": "sources", "data": []}) + "\n"
                return

            yield json.dumps({"type": "status", "message": f"Found {len(search_results)} sources, processing..."}) + "\n"

            chunks = []
            for sr in search_results:
                text = sr.content or sr.snippet
                if text:
                    chunks.append({"text": text, "title": sr.title, "url": sr.url, "snippet": sr.snippet, "source": sr.source})

            yield json.dumps({"type": "status", "message": f"Scoring {len(chunks)} results..."}) + "\n"
            scored = self.reranker.rank_with_scores(question, chunks, k=k + 2)

            if not scored or max(s for _, s in scored) < 0.1:
                yield json.dumps({"type": "status", "message": "Search results not great, using my knowledge..."}) + "\n"
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                answer_parts = []
                yield json.dumps({"type": "start"}) + "\n"
                for token in self.llm.stream(prompt):
                    answer_parts.append(token)
                    yield json.dumps({"type": "token", "data": token}) + "\n"
                answer = _parse_cited_answer("".join(answer_parts))
                answer = format_response(answer, question)
                self._extract_memories(question, answer, user_id)
                yield json.dumps({"type": "done", "data": answer}) + "\n"
                yield json.dumps({"type": "sources", "data": []}) + "\n"
                return

            reranked = [doc for doc, _ in scored[:k]]
            reranked = _clean_context(reranked)

            yield json.dumps({"type": "status", "message": "Compressing context..."}) + "\n"
            compressed = self.context_compressor.compress(reranked, question)

            yield json.dumps({"type": "status", "message": "Generating answer..."}) + "\n"

            context = _format_context_with_citations(compressed)
            if len(context) > _MAX_CONTEXT_CHARS:
                context = context[:_MAX_CONTEXT_CHARS] + "..."

            prompt = LIVE_PROMPT_TEMPLATE.format(context=context, question=question)
            prompt = self._inject_memories(question, prompt, user_id)

            answer_parts = []
            yield json.dumps({"type": "start"}) + "\n"
            for token in self.llm.stream(prompt):
                answer_parts.append(token)
                yield json.dumps({"type": "token", "data": token}) + "\n"

            answer = _parse_cited_answer("".join(answer_parts))
            answer = format_response(answer, question)
            self._extract_memories(question, answer, user_id)
            yield json.dumps({"type": "done", "data": answer}) + "\n"

            _, cited_sources = self._generate_cited_answer(prompt, compressed)
            source_list = []
            for s in enrich_sources_with_scoring(cited_sources):
                source_list.append({
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", s.get("text", ""))[:200],
                    "source": s.get("source", "web"),
                    "relevance_score": s.get("relevance_score", round(s.get("_relevance_score", 0), 4)),
                    "confidence_score": s.get("confidence_score", round(s.get("_confidence_score", 0), 4)),
                    "source_type": s.get("source_type", s.get("source", "web")),
                    "highlight_keywords": s.get("highlight_keywords", []),
                })

            yield json.dumps({"type": "sources", "data": source_list}) + "\n"

            related = self._generate_related_questions(question, answer)
            if related:
                yield json.dumps({"type": "related", "data": related}) + "\n"

            self._cache_set(question, {"answer": answer, "sources": source_list, "related_questions": related, "live": True, "cached": False, "_ts": time.time()})
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    def stream_query(self, question: str, user_id: int | None = None) -> Generator[str, None, None]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        try:
            self._ensure_llm()
            qi_result = self.query_intelligence.process(question)
            if is_fast_path(question):
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                for token in self.llm.stream(prompt):
                    yield token
                return
            decision = self.tool_decider.decide(question)
            if decision.use_tools and decision.tool in ("kb_search", "file_search"):
                search_q = decision.rewritten_query or qi_result.rewritten or question
                prompt, docs = self._build_prompt(search_q)
                prompt = self._inject_memories(question, prompt, user_id)
                for token in self.llm.stream(prompt):
                    yield token
                yield "\n__SOURCES__\n"
                for d in docs:
                    yield d + "\n---\n"
            else:
                prompt = select_prompt(question).format(question=question)
                prompt = self._inject_memories(question, prompt, user_id)
                for token in self.llm.stream(prompt):
                    yield token
        except LLMError as e:
            yield f"LLM Error: {e}"
        except Exception as e:
            yield f"Error: {e}"
