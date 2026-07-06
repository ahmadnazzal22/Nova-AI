import json
import re
import time
import uuid
from typing import Generator, Any

from ..logger import get_logger, set_correlation_id, log_pipeline
from ..query_intelligence import QueryIntelligence
from ..context_compressor import ContextCompressor
from ..ranking.reranker import enrich_sources_with_scoring

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

_NO_CONTEXT_PROMPT_TEMPLATE = """You are a professional AI assistant. Answer the user's question directly using your knowledge.

MANDATORY RULES:
- Answer the question directly. Do NOT ask if the user needs help.
- NEVER say "Is there something I can help you with?" or similar openers.
- Never repeat the user's question at the start.
- Be concise. Maximum 3-4 paragraphs. No repetition. No redundant headers.
- Start with a short introduction (1-2 sentences).
- Use ## for main headings and ### for subheadings.
- Use paragraphs, bullet points (-), and numbered lists (1.).
- Use the same language as the user.
- Keep paragraphs short (2-4 sentences).
- If the question is unclear, state what you understand and ask for clarification.

Question:
{question}"""

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


def _clean_text(text: str) -> str:
    text = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", text)
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _format_context_with_citations(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", chunk.get("content", chunk.get("snippet", "")))
        parts.append(f"[{i}] {text}")
    return "\n\n".join(parts)


def _build_citation_list(chunks: list[dict]) -> list[dict]:
    """Build a structured citation list from compressed chunks."""
    citations = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", chunk.get("content", chunk.get("snippet", "")))
        citations.append({
            "id": i,
            "text": text[:500],
            "title": chunk.get("title", ""),
            "url": chunk.get("url", ""),
            "source": chunk.get("source_type", chunk.get("source", "kb")),
            "relevance_score": round(chunk.get("relevance_score", chunk.get("_relevance_score", 0)), 4),
            "confidence_score": round(chunk.get("confidence_score", chunk.get("_confidence_score", 0)), 4),
        })
    return citations


def _parse_cited_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer).strip()


def _clean_context(chunks: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for c in chunks:
        text = c.get("text", c.get("content", c.get("snippet", "")))
        h = hash(text[:200].strip().lower())
        if h in seen:
            continue
        seen.add(h)
        cleaned = re.sub(r"\[(?!\d+\])[^\]]*\]", "", text)
        cleaned = re.sub(r"(?i)(click here|subscribe|newsletter|advertisement|sponsored|cookie policy|privacy policy)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        c["text"] = cleaned
        result.append(c)
    return result


class RAGOrchestrator:
    def __init__(self, llm_service=None, cache_service=None, vector_store=None, reranker_service=None, web_search=None, memory_service=None):
        self._llm = llm_service
        self._cache = cache_service
        self._vector_store = vector_store
        self._reranker = reranker_service
        self._web_search = web_search
        self._memory = memory_service
        self._query_intelligence = QueryIntelligence()
        self._context_compressor = ContextCompressor()

    def _get_llm(self):
        if self._llm is None:
            from ..llm.llm_service import get_llm_service
            self._llm = get_llm_service()
        self._query_intelligence.set_llm(self._llm)
        return self._llm

    def _get_cache(self):
        if self._cache is None:
            from ..cache.redis_cache import get_query_cache
            self._cache = get_query_cache()
        return self._cache

    def _get_vector_store(self):
        if self._vector_store is None:
            from ..retrieval.qdrant_store import QdrantStore
            from ..custom_embeddings import TransformerEmbeddings
            embeddings = TransformerEmbeddings()
            self._vector_store = QdrantStore(embedding_fn=embeddings.embed_query)
        return self._vector_store

    def _get_reranker(self):
        if self._reranker is None:
            from ..ranking.reranker import get_reranker_service
            self._reranker = get_reranker_service()
        return self._reranker

    def _get_web_search(self):
        if self._web_search is None:
            from ..web_search import WebSearch
            self._web_search = WebSearch(max_results=5)
        return self._web_search

    def _get_memory(self):
        if self._memory is None:
            from ..context.memory_service import get_memory_service
            self._memory = get_memory_service()
        return self._memory

    def _generate_related(self, question: str, answer: str) -> list[str]:
        try:
            prompt = RELATED_QUESTIONS_PROMPT.format(question=question, answer=answer[:500])
            response = self._get_llm().invoke(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"\s*```$", "", response)
            questions = json.loads(response)
            if isinstance(questions, list):
                return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:5]
        except Exception as e:
            logger.debug("Failed to generate related: %s", e)
        return []

    def direct_answer(self, question: str, user_id: int | None = None) -> dict:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("direct", {"q": question[:60]}, logger)

        from ..response_formatter import format_response
        prompt = _NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
        prompt = self._inject_memories(question, prompt, user_id)
        answer = self._get_llm().invoke(prompt)
        answer = format_response(answer, question)
        related = self._generate_related(question, answer)
        result = {"answer": answer, "sources": [], "cached": False, "tool_used": None}
        if related:
            result["related_questions"] = related
        return result

    def _inject_memories(self, question: str, prompt: str, user_id: int | None = None) -> str:
        if user_id is None:
            return prompt
        ctx = self._get_memory().get_short_term_context(user_id)
        if ctx:
            prompt = f"{ctx}\n\n{prompt}"
        return prompt

    def _store_memories(self, question: str, answer: str, user_id: int | None = None):
        if user_id is None:
            return
        self._get_memory().add_to_short_term(user_id, "user", question)
        self._get_memory().add_to_short_term(user_id, "assistant", answer)

    def kb_query(self, question: str, user_id: int | None = None, kb_namespace: str = "") -> dict:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("kb_query", {"q": question[:60]}, logger)

        cache = self._get_cache()
        cached = __import__("asyncio").run(cache.get("query", question))
        if cached:
            log_pipeline("cache_hit", {}, logger)
            return cached

        qi_result = self._query_intelligence.process(question)
        log_pipeline("kb_query_intelligence", {
            "intent": qi_result.intent,
            "keywords": qi_result.keywords[:5],
            "multi": len(qi_result.multi_queries),
        }, logger)

        all_sources = []
        seen = set()
        vector_store = self._get_vector_store()
        search_queries = [qi_result.rewritten or question] + (qi_result.multi_queries[1:] if len(qi_result.multi_queries) > 1 else [])
        for q in search_queries:
            results = vector_store.similarity_search(q, k=3, namespace=kb_namespace)
            for r in results:
                text = _clean_text(r.get("text", ""))
                key = text[:200].strip().lower()
                if key not in seen:
                    seen.add(key)
                    r["_relevance_score"] = r.get("score", 0.5)
                    all_sources.append(r)

        if not all_sources:
            logger.info("kb_query no_sources for '%.60s' — falling back to direct LLM", question)
            result = self.direct_answer(question, user_id=user_id)
            result["tool_used"] = "kb"
        else:
            reranker = self._get_reranker()
            scored = reranker.rank_with_scores(qi_result.original, all_sources, k=5)
            top_chunks = [doc for doc, _ in scored] if scored else all_sources[:3]

            compressed = self._context_compressor.compress(top_chunks, question)

            context = self._context_compressor.format_context(compressed)
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            prompt = self._inject_memories(question, prompt, user_id)
            answer = self._get_llm().invoke(prompt)
            from ..response_validator import ResponseValidator
            from ..response_formatter import format_response
            from ..fast_path import GREETINGS
            is_greeting = question.lower().strip() in GREETINGS
            if not is_greeting:
                if not ResponseValidator.is_valid_answer(question, answer) or (ResponseValidator.low_confidence(answer) and not question.lower().startswith(("what is", "what's", "what are", "what does", "explain"))):
                    answer = self._get_llm().invoke(ResponseValidator.build_fallback_prompt(question))
            answer = format_response(answer, question)
            self._store_memories(question, answer, user_id)

            source_list = []
            for s in enrich_sources_with_scoring(compressed):
                source_list.append({
                    "text": s.get("text", s.get("content", "")),
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet") or s.get("text", "")[:200],
                    "source": s.get("source", "kb"),
                    "relevance_score": s.get("relevance_score", round(s.get("_relevance_score", 0), 4)),
                    "confidence_score": s.get("confidence_score", round(s.get("_confidence_score", 0), 4)),
                    "source_type": s.get("source_type", s.get("source", "kb")),
                    "highlight_keywords": s.get("highlight_keywords", []),
                })
            citations = _build_citation_list(compressed)
            result = {"answer": answer, "sources": source_list, "citations": citations, "cached": False, "tool_used": "kb"}

        related = self._generate_related(question, result["answer"])
        if related:
            result["related_questions"] = related
        __import__("asyncio").run(cache.set("query", question, result))
        return result

    def live_query(self, question: str, user_id: int | None = None) -> dict:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("live_query", {"q": question[:60]}, logger)

        cache = self._get_cache()
        cached = __import__("asyncio").run(cache.get("live", question))
        if cached:
            cached["cached"] = True
            return cached

        qi_result = self._query_intelligence.process(question)
        log_pipeline("live_query_intelligence", {
            "intent": qi_result.intent,
            "keywords": qi_result.keywords[:5],
        }, logger)

        from ..fast_path import is_fast_path
        if is_fast_path(question):
            self._get_memory().add_to_short_term(user_id or 0, "user", question)
            result = self.direct_answer(question, user_id)
            result["live"] = True
            return result

        search_q = qi_result.rewritten or question
        search_results = self._get_web_search().search_and_fetch(search_q, max_results=5, max_chars_per_page=2000)

        if not search_results:
            result = self.direct_answer(question, user_id)
            result["live"] = True
            return result

        chunks = []
        for sr in search_results:
            text = sr.content or sr.snippet
            if text:
                chunks.append({"text": text, "title": sr.title, "url": sr.url, "snippet": sr.snippet, "source": sr.source})

        reranker = self._get_reranker()
        scored = reranker.rank_with_scores(question, chunks, k=5)

        if not scored or max(s for _, s in scored) < 0.1:
            result = self.direct_answer(question, user_id)
            result["live"] = True
            return result

        reranked = [doc for doc, _ in scored[:3]]
        reranked = _clean_context(reranked)

        compressed = self._context_compressor.compress(reranked, question)

        context = _format_context_with_citations(compressed)
        prompt = LIVE_PROMPT_TEMPLATE.format(context=context, question=question)
        prompt = self._inject_memories(question, prompt, user_id)
        answer = self._get_llm().invoke(prompt)
        from ..response_formatter import format_response
        answer = format_response(answer, question)
        self._store_memories(question, answer, user_id)

        source_list = []
        for s in enrich_sources_with_scoring(compressed):
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
        citations = _build_citation_list(compressed)
        result = {"answer": answer, "sources": source_list, "citations": citations, "live": True, "cached": False, "tool_used": "web"}
        related = self._generate_related(question, answer)
        if related:
            result["related_questions"] = related
        __import__("asyncio").run(cache.set("live", question, result))
        return result

    def live_stream(self, question: str, user_id: int | None = None) -> Generator[str, None, None]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        log_pipeline("stream", {"q": question[:60]}, logger)

        from ..prompt_router import select_prompt
        from ..response_formatter import format_response
        from ..fast_path import is_fast_path

        qi_result = self._query_intelligence.process(question)
        log_pipeline("stream_intelligence", {"intent": qi_result.intent}, logger)

        if is_fast_path(question):
            yield json.dumps({"type": "status", "message": "Thinking..."}) + "\n"
            prompt = select_prompt(question).format(question=question)
            prompt = self._inject_memories(question, prompt, user_id)
            parts = []
            yield json.dumps({"type": "start"}) + "\n"
            for token in self._get_llm().stream(prompt):
                parts.append(token)
                yield json.dumps({"type": "token", "data": token}) + "\n"
            answer = _parse_cited_answer("".join(parts))
            answer = format_response(answer, question)
            self._store_memories(question, answer, user_id)
            yield json.dumps({"type": "done", "data": answer}) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "citations", "data": []}) + "\n"
            return

        yield json.dumps({"type": "status", "message": "Searching the web..."}) + "\n"

        search_q = qi_result.rewritten or question
        search_results = self._get_web_search().search_and_fetch(search_q, max_results=5, max_chars_per_page=2000)

        if not search_results:
            yield json.dumps({"type": "status", "message": "Answering from my knowledge..."}) + "\n"
            prompt = select_prompt(question).format(question=question)
            prompt = self._inject_memories(question, prompt, user_id)
            parts = []
            yield json.dumps({"type": "start"}) + "\n"
            for token in self._get_llm().stream(prompt):
                parts.append(token)
                yield json.dumps({"type": "token", "data": token}) + "\n"
            answer = _parse_cited_answer("".join(parts))
            answer = format_response(answer, question)
            self._store_memories(question, answer, user_id)
            yield json.dumps({"type": "done", "data": answer}) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "citations", "data": []}) + "\n"
            return

        yield json.dumps({"type": "status", "message": f"Found {len(search_results)} sources, processing..."}) + "\n"

        chunks = []
        for sr in search_results:
            text = sr.content or sr.snippet
            if text:
                chunks.append({"text": text, "title": sr.title, "url": sr.url, "snippet": sr.snippet, "source": sr.source})

        yield json.dumps({"type": "status", "message": f"Scoring {len(chunks)} results..."}) + "\n"
        scored = self._get_reranker().rank_with_scores(question, chunks, k=5)

        if not scored or max(s for _, s in scored) < 0.1:
            yield json.dumps({"type": "status", "message": "Search results not great, using my knowledge..."}) + "\n"
            prompt = select_prompt(question).format(question=question)
            prompt = self._inject_memories(question, prompt, user_id)
            parts = []
            yield json.dumps({"type": "start"}) + "\n"
            for token in self._get_llm().stream(prompt):
                parts.append(token)
                yield json.dumps({"type": "token", "data": token}) + "\n"
            answer = _parse_cited_answer("".join(parts))
            answer = format_response(answer, question)
            self._store_memories(question, answer, user_id)
            yield json.dumps({"type": "done", "data": answer}) + "\n"
            yield json.dumps({"type": "sources", "data": []}) + "\n"
            yield json.dumps({"type": "citations", "data": []}) + "\n"
            return

        reranked = [doc for doc, _ in scored[:3]]
        reranked = _clean_context(reranked)

        yield json.dumps({"type": "status", "message": "Compressing context..."}) + "\n"
        compressed = self._context_compressor.compress(reranked, question)

        yield json.dumps({"type": "status", "message": "Generating answer..."}) + "\n"
        context = _format_context_with_citations(compressed)
        prompt = LIVE_PROMPT_TEMPLATE.format(context=context, question=question)
        prompt = self._inject_memories(question, prompt, user_id)

        parts = []
        yield json.dumps({"type": "start"}) + "\n"
        for token in self._get_llm().stream(prompt):
            parts.append(token)
            yield json.dumps({"type": "token", "data": token}) + "\n"

        answer = _parse_cited_answer("".join(parts))
        answer = format_response(answer, question)
        self._store_memories(question, answer, user_id)
        yield json.dumps({"type": "done", "data": answer}) + "\n"

        source_list = []
        for s in enrich_sources_with_scoring(compressed):
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
        citations = _build_citation_list(compressed)
        yield json.dumps({"type": "sources", "data": source_list}) + "\n"
        yield json.dumps({"type": "citations", "data": citations}) + "\n"

        related = self._generate_related(question, answer)
        if related:
            yield json.dumps({"type": "related", "data": related}) + "\n"

    def stream_query(self, question: str, user_id: int | None = None, kb_namespace: str = "") -> Generator[str, None, None]:
        from ..prompt_router import select_prompt
        qi_result = self._query_intelligence.process(question)
        prompt, docs, citations = self._build_prompt(question, qi_result, kb_namespace=kb_namespace)
        logger.info("stream_query prompt_len=%d docs=%d question='%.60s'", len(prompt), len(docs), question)
        logger.info("stream_query prompt_preview='%s'", prompt[:300] if prompt else "EMPTY")
        prompt = self._inject_memories(question, prompt, user_id)
        if not prompt or not prompt.strip():
            logger.error("stream_query EMPTY prompt after injection for question='%.60s'", question)
            return
        for token in self._get_llm().stream(prompt):
            yield token
        if docs:
            yield "\n__SOURCES__\n"
            for d in docs:
                yield d + "\n---\n"
        if citations:
            yield "\n__CITATIONS__\n" + json.dumps(citations) + "\n"

    def _build_prompt(self, question: str, qi_result=None, kb_namespace: str = "") -> tuple[str, list[str], list[dict]]:
        if qi_result is None:
            qi_result = self._query_intelligence.process(question)

        search_queries = [qi_result.rewritten or question]
        if len(qi_result.multi_queries) > 1:
            search_queries.extend(qi_result.multi_queries[1:])

        all_sources = []
        seen = set()
        vector_store = self._get_vector_store()
        for q in search_queries:
            results = vector_store.similarity_search(q, k=3, namespace=kb_namespace)
            for r in results:
                text = _clean_text(r.get("text", ""))
                key = text[:200].strip().lower()
                if key not in seen:
                    seen.add(key)
                    r["_relevance_score"] = r.get("score", 0.5)
                    all_sources.append(r)
        all_sources = all_sources[:5]

        if not all_sources:
            logger.info("_build_prompt no_sources for '%.60s' — using direct prompt", question)
            prompt = _NO_CONTEXT_PROMPT_TEMPLATE.format(question=question)
            return prompt, [], []

        reranker = self._get_reranker()
        scored = reranker.rank_with_scores(question, all_sources, k=3)
        top_chunks = [doc for doc, _ in scored] if scored else all_sources[:3]
        compressed = self._context_compressor.compress(top_chunks, question)
        context = self._context_compressor.format_context(compressed)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        source_texts = []
        for c in compressed:
            t = c.get("text", c.get("content", c.get("snippet", "")))
            if t:
                source_texts.append(t)
        citations = _build_citation_list(compressed)
        return prompt, source_texts[:3], citations

    def ask(self, question: str, user_id: int | None = None, kb_namespace: str = "") -> dict:
        """Auto-route question to the best pipeline.

        Returns a dict with keys: answer, sources, citations, route, tool_used, etc.
        """
        from .query_router import route_question as _route
        from ..fast_path import is_fast_path

        if is_fast_path(question):
            result = self.direct_answer(question, user_id)
            result["route"] = "greeting"
            return result

        route = _route(question)

        if route == "live":
            result = self.live_query(question, user_id=user_id)
            result["route"] = "live"
            return result

        if route == "research":
            from ..research_agent import DeepResearchAgent
            agent = DeepResearchAgent()
            report_text = ""
            sources = []
            citations = []
            for event in agent.research(question, kb_namespace=kb_namespace):
                try:
                    p = json.loads(event) if isinstance(event, str) else event
                    if p.get("type") == "research_done":
                        report_text = p.get("report", "")
                        sources = p.get("sources", [])
                    elif p.get("type") == "research_citations":
                        citations = p.get("data", [])
                except Exception:
                    pass
            result = {
                "answer": report_text,
                "sources": sources,
                "citations": citations,
                "route": "research",
                "tool_used": "research",
                "cached": False,
            }
            related = self._generate_related(question, report_text)
            if related:
                result["related_questions"] = related
            return result

        result = self.kb_query(question, user_id=user_id, kb_namespace=kb_namespace)
        result["route"] = "rag"
        return result

    def ask_stream(self, question: str, user_id: int | None = None, kb_namespace: str = "") -> Generator[str, None, None]:
        """Auto-route question to the best pipeline with SSE streaming.

        Yields JSON-encoded SSE event strings.
        """
        from .query_router import route_question as _route
        from ..fast_path import is_fast_path

        if is_fast_path(question):
            yield json.dumps({"type": "route", "data": "greeting"}) + "\n"
            for chunk in self.live_stream(question, user_id=user_id):
                yield chunk
            return

        route = _route(question)
        yield json.dumps({"type": "route", "data": route}) + "\n"

        if route == "live":
            for chunk in self.live_stream(question, user_id=user_id):
                yield chunk
            return

        if route == "research":
            yield json.dumps({"type": "status", "message": "Starting deep research..."}) + "\n"
            from ..research_agent import DeepResearchAgent
            agent = DeepResearchAgent()
            for event in agent.research(question, kb_namespace=kb_namespace):
                yield event if event.endswith("\n") else event + "\n"
            return

        for chunk in self.stream_query(question, user_id=user_id, kb_namespace=kb_namespace):
            yield chunk


_orchestrator: RAGOrchestrator | None = None


def get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
