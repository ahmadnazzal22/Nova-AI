import json
import os
import re
import time
import uuid
from typing import Generator, Any

from .logger import get_logger, set_correlation_id

logger = get_logger(__name__)

DECOMPOSITION_PROMPT = """You are a research planning expert. Decompose the following question into 3-6 specific sub-questions that, when answered together, provide a comprehensive answer.

For each sub-question, also specify the search strategy:
- "kb" = knowledge base (factual, definitional, conceptual)
- "web" = web search (current events, news, specific data)
- "both" = search both

Return ONLY a JSON array of objects, each with keys: "question" (string), "strategy" ("kb"|"web"|"both"), "rationale" (string).

Example:
[
  {{"question": "What are the key features of X?", "strategy": "kb", "rationale": "Core concepts are well-documented"}},
  {{"question": "What are the latest developments in X?", "strategy": "web", "rationale": "Recent news required"}}
]

Question: {question}
Sub-questions:"""

REPORT_PROMPT = """You are a professional research analyst. Create a comprehensive, well-structured research report based on the collected findings.

CONTEXT:
{context}

REQUIREMENTS:
- Write in the same language as the original question.
- Structure the report with these sections:
  ## Executive Summary
  ## Detailed Findings
    (organize findings by topic, using ### subheadings)
  ## Key Takeaways
  ## Limitations

- Use ### subheadings within Detailed Findings for each major topic.
- Cite sources using [1], [2], etc. throughout the report.
- Be thorough but concise. Use bullet points for lists of facts.
- End each section naturally - do NOT add conclusion headers unless warranted.
- Integrate findings from different sources to create a coherent narrative.
- If information is conflicting, acknowledge this and present both sides.
- If information is insufficient, clearly state what remains uncertain.
- Use Markdown formatting.

Research question: {question}
Report:"""

SUMMARIZE_FINDINGS_PROMPT = """Summarize the following research findings for the sub-question below. Extract the most important facts, statistics, and insights. Be concise.

Sub-question: {sub_question}

Findings:
{findings}

Summary (2-4 sentences):"""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _dedup_sources(sources: list[dict]) -> list[dict]:
    seen_urls = set()
    seen_texts = set()
    result = []
    for s in sources:
        url = s.get("url", "")
        text = s.get("text", s.get("snippet", ""))[:200].strip().lower()
        if url and url in seen_urls:
            continue
        if text and text in seen_texts:
            continue
        if url:
            seen_urls.add(url)
        if text:
            seen_texts.add(text)
        result.append(s)
    return result


def _score_source(source: dict, query: str) -> float:
    text = (source.get("text", "") + " " + source.get("title", "") + " " + source.get("snippet", "")).lower()
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text))
    if not query_words or not text_words:
        return 0.0
    overlap = len(query_words & text_words)
    jaccard = overlap / len(query_words | text_words)
    score = min(1.0, jaccard * 2.5 + 0.1)
    if source.get("source") in ("wikipedia", "kb"):
        score = min(1.0, score + 0.1)
    if source.get("url", "").startswith("http"):
        score = min(1.0, score + 0.05)
    return round(score, 4)


class DeepResearchAgent:
    def __init__(self, llm_service=None, vector_store=None, web_search=None, reranker=None, context_compressor=None):
        self._llm = llm_service
        self._vector_store = vector_store
        self._web_search = web_search
        self._reranker = reranker
        self._context_compressor = context_compressor

    def _get_llm(self):
        if self._llm is None:
            from .llm.llm_service import get_llm_service
            self._llm = get_llm_service()
        return self._llm

    def _get_vector_store(self):
        if self._vector_store is None:
            try:
                from .retrieval.qdrant_store import QdrantStore
                from .custom_embeddings import TransformerEmbeddings
                embeddings = TransformerEmbeddings()
                self._vector_store = QdrantStore(embedding_fn=embeddings.embed_query)
            except Exception as e:
                logger.warning("QdrantStore unavailable, trying Chroma: %s", e)
                try:
                    from langchain_chroma import Chroma
                    from .custom_embeddings import TransformerEmbeddings
                    self._vector_store = Chroma(
                        persist_directory=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
                        embedding_function=TransformerEmbeddings(),
                        collection_name=os.getenv("CHROMA_COLLECTION", "langchain"),
                    )
                    # Wrap to provide similarity_search
                    class _ChromaWrapper:
                        def __init__(self, db):
                            self._db = db
                        def similarity_search(self, query, k=3):
                            docs = self._db.similarity_search_with_score(query, k=k)
                            results = []
                            for doc, score in docs:
                                results.append({
                                    "text": doc.page_content,
                                    "title": doc.metadata.get("title", ""),
                                    "url": doc.metadata.get("url", ""),
                                    "snippet": doc.page_content[:200],
                                    "score": float(score),
                                })
                            return results
                    self._vector_store = _ChromaWrapper(self._vector_store)
                except Exception as e2:
                    logger.warning("Chroma also unavailable: %s", e2)
                    class _MockVectorStore:
                        def similarity_search(self, query, k=3):
                            return []
                    self._vector_store = _MockVectorStore()
        return self._vector_store

    def _get_web_search(self):
        if self._web_search is None:
            from .web_search import WebSearch
            self._web_search = WebSearch(max_results=5)
        return self._web_search

    def _get_reranker(self):
        if self._reranker is None:
            from .ranking.reranker import get_reranker_service
            self._reranker = get_reranker_service()
        return self._reranker

    def _get_context_compressor(self):
        if self._context_compressor is None:
            from .context_compressor import ContextCompressor
            self._context_compressor = ContextCompressor()
        return self._context_compressor

    def _search_kb(self, query: str, k: int = 3, kb_namespace: str = "") -> list[dict]:
        try:
            vs = self._get_vector_store()
            results = vs.similarity_search(query, k=k, namespace=kb_namespace)
            enriched = []
            for r in results:
                text = r.get("text", r.get("content", ""))
                if text:
                    enriched.append({
                        "text": _clean_text(text),
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "snippet": r.get("snippet", "")[:200],
                        "source": "kb",
                        "source_type": "knowledge_base",
                    })
            return enriched
        except Exception as e:
            logger.warning("KB search failed for '%s': %s", query[:60], e)
            return []

    def _search_web(self, query: str, max_results: int = 3) -> list[dict]:
        try:
            ws = self._get_web_search()
            results = ws.search_and_fetch(query, max_results=max_results, max_chars_per_page=2000)
            enriched = []
            for sr in results:
                text = sr.content or sr.snippet
                if text:
                    enriched.append({
                        "text": _clean_text(text),
                        "title": sr.title,
                        "url": sr.url,
                        "snippet": sr.snippet[:200],
                        "source": sr.source,
                        "source_type": "web",
                    })
            return enriched
        except Exception as e:
            logger.warning("Web search failed for '%s': %s", query[:60], e)
            return []

    def _decompose_question(self, question: str) -> list[dict]:
        try:
            prompt = DECOMPOSITION_PROMPT.format(question=question)
            response = self._get_llm().invoke(prompt)
            response = response.strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response)
            # Extract JSON array from response (LLM may prepend text like "Here are the sub-questions:")
            start = response.find("[")
            end = response.rfind("]")
            if start != -1 and end != -1 and end > start:
                response = response[start:end+1]
            sub_questions = json.loads(response)
            if isinstance(sub_questions, list):
                validated = []
                for sq in sub_questions[:6]:
                    if isinstance(sq, dict) and sq.get("question"):
                        validated.append({
                            "question": sq["question"],
                            "strategy": sq.get("strategy", "both"),
                            "rationale": sq.get("rationale", ""),
                        })
                if validated:
                    return validated
        except Exception as e:
            logger.warning("Question decomposition failed: %s", e)
        return [{"question": question, "strategy": "both", "rationale": "Direct research"}]

    def _search_step(self, sub_q: str, strategy: str, kb_namespace: str = "") -> list[dict]:
        sources = []
        if strategy in ("kb", "both"):
            kb_results = self._search_kb(sub_q, k=3, kb_namespace=kb_namespace)
            for r in kb_results:
                r["_search_query"] = sub_q
            sources.extend(kb_results)
        if strategy in ("web", "both") and len(sources) < 3:
            web_results = self._search_web(sub_q, max_results=3)
            for r in web_results:
                r["_search_query"] = sub_q
            sources.extend(web_results)
        for s in sources:
            s["_relevance_score"] = _score_source(s, sub_q)
        sources.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
        return sources[:4]

    def _generate_summary(self, sub_question: str, findings: list[dict]) -> str:
        try:
            findings_text = "\n\n".join([
                f"[{i+1}] {s.get('text', s.get('snippet', ''))[:500]}"
                for i, s in enumerate(findings)
            ])
            prompt = SUMMARIZE_FINDINGS_PROMPT.format(sub_question=sub_question, findings=findings_text)
            return self._get_llm().invoke(prompt).strip()
        except Exception as e:
            logger.warning("Summary generation failed: %s", e)
            return ""

    def _generate_report(self, question: str, all_sources: list[dict]) -> Generator[str, None, None]:
        context_parts = []
        for i, s in enumerate(all_sources, 1):
            text = s.get("text", s.get("snippet", ""))
            title = s.get("title", "")
            source = s.get("source", "unknown")
            if text:
                context_parts.append(f"[{i}] From {title} ({source}): {text[:800]}")
        context = "\n\n".join(context_parts)
        prompt = REPORT_PROMPT.format(question=question, context=context)
        for token in self._get_llm().stream(prompt):
            yield token

    def _build_citation_list(self, sources: list[dict]) -> list[dict]:
        from .citation.citation_injector import CitationInjector
        _, citations = CitationInjector.build_context_with_citations(sources)
        return citations

    def _generate_follow_up(self, question: str, report: str) -> list[str]:
        try:
            prompt = (
                f"Based on this research Q&A, suggest 3-5 related questions the user might ask next.\n"
                f"Return ONLY a JSON array of strings.\n\n"
                f"Q: {question}\n"
                f"A: {report[:1000]}\n\n"
                f"Related questions:"
            )
            response = self._get_llm().invoke(prompt).strip()
            if response.startswith("```"):
                response = re.sub(r"^```(?:json)?\s*|\s*```$", "", response)
            questions = json.loads(response)
            if isinstance(questions, list):
                return [q.strip() for q in questions if isinstance(q, str) and q.strip()][:5]
        except Exception as e:
            logger.debug("Follow-up generation failed: %s", e)
        return []

    def research(self, question: str, kb_namespace: str = "") -> Generator[str, None, None]:
        cid = uuid.uuid4().hex[:12]
        set_correlation_id(cid)
        logger.info("[%s] Deep research: %.80s (namespace='%s')", cid, question, kb_namespace)

        try:
            # 1. Decompose question
            yield json.dumps({"type": "research_status", "message": "Analyzing question and creating research plan..."}) + "\n"
            sub_questions = self._decompose_question(question)
            yield json.dumps({
                "type": "research_plan",
                "sub_questions": [sq["question"] for sq in sub_questions],
                "total_steps": len(sub_questions),
            }) + "\n"
            logger.info("[%s] Decomposed into %d sub-questions", cid, len(sub_questions))

            # 2. Search each sub-question
            all_sources = []
            step_sources = {}
            for step_idx, sq in enumerate(sub_questions):
                step_num = step_idx + 1
                sq_question = sq["question"]
                sq_strategy = sq["strategy"]

                yield json.dumps({
                    "type": "research_step",
                    "step": step_num,
                    "total": len(sub_questions),
                    "sub_question": sq_question,
                    "strategy": sq_strategy,
                    "rationale": sq.get("rationale", ""),
                    "status": "searching",
                }) + "\n"

                logger.info("[%s] Step %d/%d: %s (strategy=%s)", cid, step_num, len(sub_questions), sq_question[:60], sq_strategy)
                sources = self._search_step(sq_question, sq_strategy, kb_namespace=kb_namespace)

                for src in sources:
                    yield json.dumps({
                        "type": "research_source",
                        "step": step_num,
                        "source": {
                            "title": src.get("title", ""),
                            "url": src.get("url", ""),
                            "snippet": src.get("snippet", src.get("text", ""))[:150],
                            "source": src.get("source", "web"),
                            "score": src.get("_relevance_score", 0),
                        },
                    }) + "\n"

                yield json.dumps({
                    "type": "research_step",
                    "step": step_num,
                    "total": len(sub_questions),
                    "sub_question": sq_question,
                    "status": "complete",
                    "sources_found": len(sources),
                }) + "\n"

                for s in sources:
                    all_sources.append(s)
                step_sources[step_num] = sources

            # 3. Deduplicate and score all sources
            all_sources = _dedup_sources(all_sources)
            for s in all_sources:
                if "_relevance_score" not in s:
                    s["_relevance_score"] = _score_source(s, question)
            all_sources.sort(key=lambda x: x.get("_relevance_score", 0), reverse=True)
            top_sources = all_sources[:15]

            if not top_sources:
                logger.info("[%s] No external sources found, using LLM knowledge", cid)
                yield json.dumps({"type": "research_status", "message": "No external sources found. Generating answer from available knowledge..."}) + "\n"
                report_parts = []
                yield json.dumps({"type": "research_start"}) + "\n"
                direct_prompt = REPORT_PROMPT.format(
                    question=question,
                    context="No external sources were found for this query. Please answer based on your general knowledge, clearly indicating which parts are well-established facts versus areas of active research or debate."
                )
                for token in self._get_llm().stream(direct_prompt):
                    report_parts.append(token)
                    yield json.dumps({"type": "research_token", "token": token}) + "\n"
                report = "".join(report_parts)
                confidence = 0.35
                source_list = []
                follow_up = self._generate_follow_up(question, report)
                yield json.dumps({
                    "type": "research_done",
                    "report": report,
                    "sources": source_list,
                    "follow_up": follow_up,
                    "confidence": confidence,
                    "total_sources": 0,
                    "total_steps": len(sub_questions),
                }) + "\n"
                yield json.dumps({"type": "research_citations", "data": []}) + "\n"
                logger.info("[%s] Research complete (LLM knowledge fallback)", cid)
                return

            # 4. Generate report
            yield json.dumps({"type": "research_status", "message": f"Synthesizing findings from {len(top_sources)} sources..."}) + "\n"

            report_parts = []
            yield json.dumps({"type": "research_start"}) + "\n"
            for token in self._generate_report(question, top_sources):
                report_parts.append(token)
                yield json.dumps({"type": "research_token", "token": token}) + "\n"

            report = "".join(report_parts)

            # 5. Generate follow-up questions
            yield json.dumps({"type": "research_status", "message": "Generating follow-up questions..."}) + "\n"
            follow_up = self._generate_follow_up(question, report)

            # 6. Compute confidence score
            scores = [s.get("_relevance_score", 0) for s in top_sources]
            avg_score = sum(scores) / len(scores) if scores else 0
            confidence = round(min(1.0, avg_score * 1.5 + 0.2), 2)

            # 7. Build source list for output
            source_list = []
            for s in top_sources:
                source_list.append({
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", s.get("text", ""))[:200],
                    "source": s.get("source", "web"),
                    "score": s.get("_relevance_score", 0),
                })

            research_citations = self._build_citation_list(top_sources)

            yield json.dumps({
                "type": "research_done",
                "report": report,
                "sources": source_list,
                "follow_up": follow_up,
                "confidence": confidence,
                "total_sources": len(top_sources),
                "total_steps": len(sub_questions),
            }) + "\n"
            yield json.dumps({"type": "research_citations", "data": research_citations}) + "\n"

            logger.info("[%s] Research complete: %d sources, confidence=%.2f", cid, len(top_sources), confidence)

        except Exception as e:
            logger.error("[%s] Research failed: %s", cid, e)
            yield json.dumps({"type": "research_error", "error": str(e)}) + "\n"
