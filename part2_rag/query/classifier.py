from typing import Literal
from dataclasses import dataclass

QueryIntent = Literal["GREETING", "HELP", "SPORTS", "SIMPLE_LLM", "COMPLEX_RAG", "CODE", "LIVE_SEARCH"]

_GREETINGS = frozenset({
    "hi", "hello", "hey", "هاي", "مرحبا", "السلام عليكم",
    "thanks", "thank you", "thankyou", "bye", "goodbye",
    "ok", "okay", "yes", "no", "sure", "great", "nice",
})

_SPORTS_KEYWORDS = frozenset({
    "مباراة", "مباريات", "نتيجة", "موعد المباراة", "كرة القدم",
    "match", "score", "game today", "fixture",
})

_CODE_STARTS = (
    "write ", "write code", "write a", "write an",
    "python ", "javascript", "typescript", "java ", "rust ",
    "go lang", "coding", "debug", "debugging",
    "function ", "class ", "implement",
)

_REAL_TIME_WORDS = frozenset({
    "latest", "news", "today", "current", "weather",
    "temperature", "price", "stock", "bitcoin", "ethereum",
    "crypto", "forecast", "humidity",
})

_SIMPLE_QUERY_MAX_WORDS = 4

_EXPLANATION_STARTS = (
    "explain", "describe", "define", "what is", "what's",
    "what are", "what does", "how does", "how do", "how to",
    "why is", "why does", "why do",
)


@dataclass
class ClassifierResult:
    intent: QueryIntent
    confidence: float
    rewritten_query: str = ""
    reason: str = ""


def classify_query(query: str) -> ClassifierResult:
    q = query.strip().lower()
    if not q:
        return ClassifierResult(intent="GREETING", confidence=1.0, reason="Empty query")

    if q in _GREETINGS:
        return ClassifierResult(intent="GREETING", confidence=1.0, reason="Greeting detected")

    if "help" in q or "مساعدة" in q:
        return ClassifierResult(intent="HELP", confidence=0.9, reason="Help keyword")

    if any(kw in q for kw in _SPORTS_KEYWORDS):
        return ClassifierResult(intent="SPORTS", confidence=0.9, reason="Sports keyword")

    words = q.split()
    if any(q.startswith(start) for start in _CODE_STARTS):
        return ClassifierResult(intent="CODE", confidence=0.8, reason="Code-related query")

    word_set = set(words)
    has_real_time = bool(_REAL_TIME_WORDS & word_set)
    is_explanation = q.startswith(_EXPLANATION_STARTS)

    if has_real_time:
        return ClassifierResult(intent="LIVE_SEARCH", confidence=0.8, reason="Real-time data needed")

    if is_explanation and len(words) > 2:
        return ClassifierResult(intent="COMPLEX_RAG", confidence=0.65, reason="Explanation query needs knowledge base")

    if len(words) <= _SIMPLE_QUERY_MAX_WORDS and not has_real_time:
        return ClassifierResult(intent="SIMPLE_LLM", confidence=0.7, reason="Short query, likely simple")

    return ClassifierResult(intent="COMPLEX_RAG", confidence=0.6, reason="Complex query requires retrieval")
