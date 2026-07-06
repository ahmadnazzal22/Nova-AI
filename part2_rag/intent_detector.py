import re
from typing import Any
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)

_GENERAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|howdy|greetings)(\s|$)", re.I),
    re.compile(r"^(good\s+)?(morning|afternoon|evening)", re.I),
    re.compile(r"^(how\s+are\s+you|how('s| is) it going|what's up|sup)", re.I),
    re.compile(r"^(who\s+are\s+you|what\s+are\s+you)", re.I),
    re.compile(r"^(what\s+can\s+you\s+do|how\s+do\s+you\s+work|what's your purpose)", re.I),
    re.compile(r"^(tell\s+(me\s+)?(a\s+)?(joke|story|fun\s+fact))", re.I),
    re.compile(r"^(thanks?|thank\s+you|appreciate\s+it)", re.I),
    re.compile(r"^(bye|goodbye|see\s+you|farewell)", re.I),
    re.compile(r"^(yes|no|ok|okay|sure|alright)", re.I),
    re.compile(r"^(how\s+do\s+(i|you)|can\s+you)"),  # ambiguous — handled by LLM fallback
]

_GENERAL_KEYWORD_ANY = {
    "your name", "nice to meet", "pleasure", "how do you feel",
    "do you like", "what's your favorite", "do you have feelings",
    "are you sentient", "are you conscious",
}

_RETRIEVAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(what\s+is|what\s+are|what's)\s+(a|an|the|this|that)\b", re.I),
    re.compile(r"^(how\s+does|how\s+do|how\s+is|how\s+are)\s", re.I),
    re.compile(r"^(why\s+(is|are|do|does|did|can|would|could))", re.I),
    re.compile(r"^(who\s+(is|are|was|were|invented|created|discovered))", re.I),
    re.compile(r"^(when\s+(was|were|did|is|are|will))", re.I),
    re.compile(r"^(explain|describe|define|compare|contrast|summarize|list)\s", re.I),
    re.compile(r"^(what's the difference|difference between)", re.I),
]


class QueryIntent(Enum):
    GENERAL = "general"
    RETRIEVAL = "retrieval"


INTENT_PROMPT = """You are a classifier. Determine if the user question needs to search a knowledge base or the web.

Answer ONLY with one word: "search" if the question asks for factual data, definitions, explanations, technical information, or anything requiring external sources.
Answer ONLY with one word: "chat" if the question is casual conversation, greeting, opinion, joke, thanks, farewell, or anything about yourself.

Question: {question}

Answer (one word, "search" or "chat"):"""


# ── Fast classification ───────────────────────────────────────────────

def _classify_fast(question: str) -> QueryIntent | None:
    q = question.strip()
    if not q:
        return QueryIntent.GENERAL

    q_lower = q.lower()

    for pattern in _GENERAL_PATTERNS:
        if pattern.match(q):
            return QueryIntent.GENERAL

    for kw in _GENERAL_KEYWORD_ANY:
        if kw in q_lower:
            return QueryIntent.GENERAL

    for pattern in _RETRIEVAL_PATTERNS:
        if pattern.match(q):
            return QueryIntent.RETRIEVAL

    return None


# ── Intent Detector ───────────────────────────────────────────────────

class IntentDetector:
    def __init__(self, llm: Any | None = None):
        self.llm = llm
        self._llm_available = llm is not None

    def set_llm(self, llm: Any) -> None:
        self.llm = llm
        self._llm_available = True

    def classify(self, question: str) -> QueryIntent:
        fast = _classify_fast(question)
        if fast is not None:
            logger.debug("Intent fast-classified: %s -> %s", question[:40], fast.value)
            return fast

        if not self._llm_available:
            logger.debug("No LLM available for intent, defaulting to RETRIEVAL: %.40s", question)
            return QueryIntent.RETRIEVAL

        return self._classify_with_llm(question)

    def _classify_with_llm(self, question: str) -> QueryIntent:
        try:
            prompt = INTENT_PROMPT.format(question=question)
            answer = self.llm.invoke(prompt).strip().lower()
            if "chat" in answer and "search" not in answer:
                logger.debug("Intent LLM-classified -> GENERAL: %.40s (answer=%s)", question, answer)
                return QueryIntent.GENERAL
            logger.debug("Intent LLM-classified -> RETRIEVAL: %.40s (answer=%s)", question, answer)
            return QueryIntent.RETRIEVAL
        except Exception as e:
            logger.warning("LLM intent classification failed, defaulting to RETRIEVAL: %s", e)
            return QueryIntent.RETRIEVAL
