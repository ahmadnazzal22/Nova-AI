import os
import re
import json
from typing import Any
from dataclasses import dataclass, field

from .logger import get_logger

logger = get_logger(__name__)


_INTENT_PATTERNS: dict[str, list[str]] = {
    "factual": [
        "what is", "what are", "what was", "what does", "who is",
        "when did", "where is", "which", "define", "definition",
        "fact", "tell me about", "explain", "describe",
    ],
    "reasoning": [
        "why", "how does", "how do", "how can", "how would",
        "what if", "what would", "what causes", "explain why",
        "compare", "difference", "versus", "vs", "pros and cons",
        "analysis", "analyze", "evaluate",
    ],
    "code": [
        "write code", "code for", "implement", "function",
        "python", "javascript", "programming", "algorithm",
        "bug", "error", "debug", "syntax", "script",
        "how to code", "example code",
    ],
    "summarization": [
        "summarize", "summary", "tl;dr", "recap", "brief",
        "in short", "condense", "overview", "key points",
    ],
    "comparison": [
        "compare", "difference", "versus", "vs", "better",
        "pros and cons", "advantages and disadvantages",
        "similarities", "compare and contrast",
    ],
    "list": [
        "list", "list of", "types of", "examples of",
        "top", "best", "enumeration", "categorize",
    ],
    "steps": [
        "steps", "step by step", "how to", "guide",
        "tutorial", "instructions", "procedure",
    ],
}

_MULTI_QUERY_TEMPLATES: dict[str, list[str]] = {
    "factual": [
        "{question}",
        "What are the key facts about {keywords}?",
        "Define and explain {keywords} in detail",
    ],
    "reasoning": [
        "{question}",
        "Explain why {keywords} works the way it does",
        "What are the underlying principles of {keywords}?",
    ],
    "code": [
        "{question}",
        "Show example code for {keywords}",
        "What are best practices for implementing {keywords}?",
    ],
    "summarization": [
        "{question}",
        "Key points and main ideas about {keywords}",
        "Brief overview of {keywords}",
    ],
}


@dataclass
class QueryIntelligenceResult:
    original: str = ""
    rewritten: str = ""
    intent: str = "factual"
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    multi_queries: list[str] = field(default_factory=list)
    confidence: float = 1.0


_STOP_WORDS: set[str] = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "it's",
    "and", "or", "but", "not", "no", "nor", "so", "if", "then",
    "than", "too", "very", "just", "about", "also", "more",
    "some", "any", "each", "every", "all", "both", "few", "most",
    "what", "which", "who", "whom", "when", "where", "why", "how",
}

_ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b"
)

_CONSTRAINT_WORDS = frozenset({
    "in", "within", "between", "after", "before", "during",
    "using", "with", "without", "through", "via",
    "for", "from", "since", "until", "except", "including",
    "above", "below", "less than", "more than", "at least",
    "at most", "maximum", "minimum", "latest", "recent",
    "specifically", "particularly", "especially",
})


def _extract_all_keywords(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]\w+(?:[-/]\w+)*", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOP_WORDS]


def _extract_entities(text: str) -> list[str]:
    return list(set(_ENTITY_PATTERN.findall(text)))


def _extract_constraints(text: str) -> list[str]:
    lower = text.lower()
    tokens = lower.split()
    constraints = []
    for i, token in enumerate(tokens):
        if token in _CONSTRAINT_WORDS and i + 1 < len(tokens):
            phrase = " ".join(tokens[i:i+3])
            if len(phrase) > 3:
                constraints.append(phrase)
    return constraints[:5]


def _rule_intent(question: str) -> tuple[str, float]:
    q = question.lower().strip()
    scores: dict[str, int] = {}
    for intent, patterns in _INTENT_PATTERNS.items():
        count = 0
        for pat in patterns:
            if pat in q:
                count += 1
        if count:
            scores[intent] = count
    if not scores:
        return "factual", 0.5
    ordered = ["code", "steps", "comparison", "reasoning", "summarization", "list", "factual"]
    best = max(ordered, key=lambda i: scores.get(i, 0))
    return best, min(1.0, scores.get(best, 0) * 0.3 + 0.3)


_INTENT_QUERY_TEMPLATES: dict[str, str] = {
    "factual": (
        "Given the question, return a rewritten version that is clear, "
        "specific, and optimized for information retrieval.\n"
        "Fix typos, expand abbreviations, and make it self-contained.\n"
        "Return ONLY the rewritten question, nothing else.\n"
        "Question: {question}\n"
        "Rewritten:"
    ),
    "reasoning": (
        "Given this question that requires reasoning or analysis, "
        "rewrite it to clarify what specific aspects need to be analyzed.\n"
        "Return ONLY the rewritten question.\n"
        "Question: {question}\n"
        "Rewritten:"
    ),
    "code": (
        "Given this programming question, rewrite it to include "
        "the programming language and specific requirements.\n"
        "Return ONLY the rewritten question.\n"
        "Question: {question}\n"
        "Rewritten:"
    ),
}


class QueryIntelligence:
    def __init__(self, llm: Any = None, embeddings: Any = None):
        self._llm = llm
        self._embeddings = embeddings
        self._cache: dict[str, QueryIntelligenceResult] = {}

    def set_llm(self, llm: Any):
        self._llm = llm

    def process(self, question: str) -> QueryIntelligenceResult:
        if question in self._cache:
            return self._cache[question]

        result = QueryIntelligenceResult(original=question)

        result.intent, result.confidence = _rule_intent(question)

        result.keywords = _extract_all_keywords(question)
        result.entities = _extract_entities(question)
        result.constraints = _extract_constraints(question)

        result.rewritten = self._llm_rewrite(question, result.intent)

        result.multi_queries = self._generate_multi_queries(
            question, result.keywords, result.intent
        )

        result.keywords = result.keywords[:10]
        result.entities = result.entities[:5]

        self._cache[question] = result
        if len(self._cache) > 200:
            self._cache.clear()

        logger.debug(
            "QueryIntelligence: intent=%s confidence=%.2f keywords=%d entities=%d multi=%d",
            result.intent, result.confidence, len(result.keywords),
            len(result.entities), len(result.multi_queries),
        )
        return result

    def _llm_rewrite(self, question: str, intent: str) -> str:
        template = _INTENT_QUERY_TEMPLATES.get(intent, _INTENT_QUERY_TEMPLATES["factual"])
        if self._llm is None:
            return question
        try:
            prompt = template.format(question=question)
            rewritten = self._llm.invoke(prompt)
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten and 10 < len(rewritten) < len(question) * 3 and rewritten.lower() != question.lower():
                return rewritten
        except Exception as e:
            logger.debug("LLM rewrite failed: %s", e)
        return question

    def _generate_multi_queries(
        self, question: str, keywords: list[str], intent: str
    ) -> list[str]:
        queries: list[str] = [question]
        templates = _MULTI_QUERY_TEMPLATES.get(intent, _MULTI_QUERY_TEMPLATES["factual"])
        kw_str = " ".join(keywords[:5]) if keywords else question
        for tmpl in templates[1:]:
            try:
                q = tmpl.format(question=question, keywords=kw_str)
                if q.lower() != question.lower():
                    queries.append(q)
            except Exception:
                pass

        if self._llm is not None and len(keywords) >= 2:
            try:
                llm_prompt = (
                    f"Generate 2 alternative search queries for this question. "
                    f"Each should be a different angle or phrasing.\n"
                    f"Return ONLY a JSON array of strings.\n"
                    f"Question: {question}\n"
                    f"Queries:"
                )
                resp = self._llm.invoke(llm_prompt)
                resp = resp.strip()
                if resp.startswith("```"):
                    resp = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp)
                extra = json.loads(resp)
                if isinstance(extra, list):
                    for q in extra[:3]:
                        if isinstance(q, str) and q.strip() and q.lower() != question.lower():
                            queries.append(q.strip())
            except Exception as e:
                logger.debug("LLM multi-query generation failed: %s", e)

        seen: set[str] = set()
        deduped = []
        for q in queries:
            key = q.lower().strip()
            if key not in seen:
                seen.add(key)
                deduped.append(q)
        return deduped[:5]
