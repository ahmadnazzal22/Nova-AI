import re
from typing import Literal

Route = Literal["greeting", "live", "research", "rag"]

LIVE_PATTERNS = re.compile(
    r"(?i)\b(?:"
    r"current|latest|today|this week|this month|breaking|just now|"
    r"news|headlines|weather|forecast|temperature|"
    r"price|stock|market|bitcoin|ethereum|crypto|"
    r"sport|score|match|game|highlight|"
    r"election|poll|president|"
    r"release|announce|launch|"
    r"update|status|reportedly"
    r")\b"
)

LIVE_QUESTION_PATTERNS = re.compile(
    r"(?i)^(?:"
    r"what('s| is).* (?:news|weather|price|score|stock|market|election|forecast|temperature)|"
    r"(?:who|how).* (?:currently|recently|yesterday|today|this week)|"
    r"(?:is|are|did|does|has|have) .* (?:just|yet|already|now)|"
    r"how much (?:is|are)|"
    r"tell me (?:something|more) about .* (?:today|now|current)|"
    r".*\b(?:breaking news|latest development|just happened)\b"
    r")"
)

COMPLEX_PATTERNS = re.compile(
    r"(?i)\b(?:"
    r"compare|contrast|difference between|differences between|similarities|"
    r"analyze|analysis|evaluate|assessment|"
    r"explain why|explain how|how does.*work|"
    r"what are the (?:key|main|primary|critical|essential)|"
    r"pros and cons|advantages and disadvantages|"
    r"impact|implication|consequence|"
    r"relationship between|correlation between|"
    r"comprehensive|in-depth|detailed|thorough|"
    r"step by step|process of|mechanism of|"
    r"the future of|long-term|short-term|"
    r"trend|pattern|cause|effect|factor"
    r")\b"
)


def route_question(question: str) -> Route:
    """Classify a question into one of the routing buckets.

    Priority order:
      1. greeting →  RAG (handled by is_fast_path upstream)
      2. live      →  Live Search
      3. complex   →  Deep Research
      4. other     →  RAG
    """
    q = question.strip()
    if not q:
        return "rag"

    # Check live patterns first (they are most specific)
    if LIVE_QUESTION_PATTERNS.match(q) or LIVE_PATTERNS.search(q):
        return "live"

    # Check complex patterns (at least 3 words to avoid trivial hits)
    if len(q.split()) >= 3 and COMPLEX_PATTERNS.search(q):
        return "research"

    return "rag"


def needs_live_search(question: str) -> bool:
    """Quick check: does this question need a live web search?"""
    return route_question(question) == "live"


def needs_deep_research(question: str) -> bool:
    """Quick check: does this question need deep research?"""
    return route_question(question) == "research"
